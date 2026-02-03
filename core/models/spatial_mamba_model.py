"""
空间感知事件Mamba主模型 - 重构版
核心主干模型，保持简洁和高效
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

from .mamba_blocks import StackedStatefulMambaBlocks
from .spatial_encoder import SpatialKNNEncoder


class SpatialAwareEventMamba(nn.Module):
    """
    空间感知事件Mamba模型 - 核心版本
    专注于核心功能，避免过度复杂化
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        # 空间编码器配置
        k_neighbors: int = 16,
        spatial_radius: Optional[float] = 50.0,
        causal_knn: bool = True,
        # Mamba配置
        mamba_blocks: int = 1,
        state_dim: int = 32,
        expand: int = 2,
        # 其他配置
        window_size: int = 4,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # 基础属性
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.window_size = window_size
        
        # ====== 1. 空间编码器 ======
        self.spatial_encoder = SpatialKNNEncoder(
            input_dim=input_dim,
            output_dim=hidden_dim,
            k_neighbors=k_neighbors,
            spatial_radius=spatial_radius,
            aggregation='attention',
            dropout=dropout_rate,
            causal=causal_knn
        )
        
        # ====== 2. 时序建模 ======
        # self.temporal_blocks = StackedStatefulMambaBlocks(
        #     num_blocks=mamba_blocks,
        #     d_model=hidden_dim,
        #     d_state=state_dim,
        #     d_conv=4,
        #     expand=expand,
        #     dropout=dropout_rate,
        # )

        self.temporal_blocks = StackedStatefulMambaBlocks(
            num_blocks=mamba_blocks,
            d_model=hidden_dim,
            d_state=state_dim,
            d_conv=4,
            expand=expand,
            dropout=dropout_rate,
        )
        
        # ====== 3. 分类头 ======
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # ====== 4. 状态管理 ======
        self._states = None
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        frame_indices: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        chunk_idx: Optional[torch.Tensor] = None,
        hist_len: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """前向传播 - 历史部分只用于空间编码，不参与时序建模和分类"""
        B, N, _ = features.shape
        
        # Step 1: 空间编码（使用完整序列）
        enhanced_features = self.spatial_encoder(features, coords, valid_mask, chunk_idx, hist_len)
        
        # Step 2: 只对当前块做时序建模
        if hist_len is not None:
            # 分离历史和当前块
            current_features = []
            for b in range(B):
                h_len = hist_len[b].item()
                current_feat = enhanced_features[b, h_len:]
                current_features.append(current_feat)
            
            # 填充到相同长度
            max_len = max(f.size(0) for f in current_features)
            padded_features = []
            for f in current_features:
                if f.size(0) < max_len:
                    padding = torch.zeros(max_len - f.size(0), f.size(1), 
                                        device=f.device, dtype=f.dtype)
                    f = torch.cat([f, padding], dim=0)
                padded_features.append(f)
            temporal_input = torch.stack(padded_features, dim=0)
        else:
            temporal_input = enhanced_features
        
        # Step 3: 时序建模
        temporal_features = self.temporal_blocks.forward(temporal_input)
        
        # Step 4: 分类
        logits = self.classifier(temporal_features)
        
        # Step 5: 处理历史对齐
        if hist_len is not None:
            full_logits = []
            for b in range(B):
                h_len = hist_len[b].item()
                if h_len > 0:
                    # 历史部分用dummy logits
                    hist_logits = torch.zeros(h_len, self.num_classes, 
                                            device=logits.device, dtype=logits.dtype)
                    curr_logits = logits[b, :N-h_len]
                    full_logit = torch.cat([hist_logits, curr_logits], dim=0)
                else:
                    full_logit = logits[b]
                
                # 确保长度匹配
                if full_logit.size(0) < N:
                    padding = torch.zeros(N - full_logit.size(0), self.num_classes,
                                        device=logits.device, dtype=logits.dtype)
                    full_logit = torch.cat([full_logit, padding], dim=0)
                full_logits.append(full_logit[:N])
            
            logits = torch.stack(full_logits, dim=0)
        
        return {
            'logits': logits,
            'features': temporal_features
        }
    
    def reset_stream(self, batch_size: int = 1, device: Optional[torch.device] = None):
        """重置流式处理状态"""
        if device is None:
            device = next(self.parameters()).device
        
        self._states = self.temporal_blocks.init_states(batch_size, device)
    
    def step_block(
        self,
        features_blk: torch.Tensor,
        coords_blk: torch.Tensor,
        knn_ctx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        block_len: int = 1
    ) -> torch.Tensor:
        """流式分块处理"""
        B, Lb, _ = features_blk.shape
        assert B == 1, "Streaming mode only supports batch_size=1"
        
        # Step 1: 空间编码（使用历史上下文）
        if knn_ctx is not None:
            features_hist, coords_hist = knn_ctx
            features_all = torch.cat([features_hist, features_blk], dim=1)
            coords_all = torch.cat([coords_hist, coords_blk], dim=1)
            
            enhanced_all = self.spatial_encoder(features_all, coords_all)
            enhanced_blk = enhanced_all[:, -Lb:]
        else:
            enhanced_blk = self.spatial_encoder(features_blk, coords_blk)
        
        # Step 2: 时序建模
        if self._states is None:
            self._states = self.temporal_blocks.init_states(batch_size=1, device=enhanced_blk.device)
        
        temporal_blk, self._states = self.temporal_blocks.forward_chunk(
            enhanced_blk, self._states, block_len=block_len
        )
        
        # Step 3: 分类
        logits_blk = self.classifier(temporal_blk)
        
        return logits_blk.squeeze(0)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置构建模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        构建好的模型
    """
    model_type = config.get('type', 'SpatialAwareEventMamba')
    
    if model_type == 'SpatialAwareEventMamba':
        return SpatialAwareEventMamba(**config)
    elif model_type == 'AblationEventModel':
        # 导入消融实验模型
        from ablation.models.ablation_model import AblationEventModel
        return AblationEventModel(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")