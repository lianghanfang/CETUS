"""
KNN聚合方法变体 - 消融实验用
专注于不同的邻居特征聚合策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class KNNAggregationVariants(nn.Module):
    """KNN聚合方法变体基类"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k_neighbors: int = 16,
        spatial_radius: Optional[float] = None,
        aggregation: str = 'mean',
        dropout: float = 0.1,
        causal: bool = True,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_neighbors = k_neighbors
        self.spatial_radius = spatial_radius
        self.aggregation = aggregation
        self.causal = causal
        
        # 特征投影
        self.feat_proj = nn.Linear(input_dim, output_dim)
        
        # 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        # 聚合相关层
        self.setup_aggregation_layers(aggregation, output_dim, dropout)
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def setup_aggregation_layers(self, aggregation: str, output_dim: int, dropout: float):
        """设置聚合相关的层"""
        if aggregation == 'attention':
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.query_proj = nn.Linear(output_dim, output_dim)
            self.key_proj = nn.Linear(output_dim, output_dim)
            self.value_proj = nn.Linear(output_dim, output_dim)
        elif aggregation == 'weighted_mean':
            self.weight_net = nn.Sequential(
                nn.Linear(output_dim, output_dim // 2),
                nn.ReLU(),
                nn.Linear(output_dim // 2, 1),
                nn.Sigmoid()
            )
        elif aggregation == 'gated':
            self.gate_net = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid()
            )
    
    def find_knn(
        self, 
        coords: torch.Tensor, 
        k: int, 
        causal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """找K近邻 - 基础实现"""
        N = coords.shape[0]
        k = min(k, N)
        
        # 计算距离矩阵
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        spatial_dist = torch.norm(diff[..., :2], dim=-1)
        temporal_dist = torch.abs(diff[..., 2])
        distances = spatial_dist + 0.3 * temporal_dist
        
        # 排除自循环
        distances.fill_diagonal_(float('inf'))
        
        # 因果屏蔽
        if causal:
            t = coords[:, 2]
            future_mask = t.unsqueeze(0) > t.unsqueeze(1)
            distances.masked_fill_(future_mask, float('inf'))
        
        # 空间半径过滤
        if self.spatial_radius is not None:
            distances = torch.where(
                spatial_dist <= self.spatial_radius,
                distances,
                float('inf')
            )
        
        # 找K近邻
        knn_dist, knn_idx = torch.topk(distances, k, dim=1, largest=False)
        valid_mask = torch.isfinite(knn_dist)
        
        knn_dist = torch.where(valid_mask, knn_dist, torch.zeros_like(knn_dist))
        
        return knn_idx, knn_dist, valid_mask
    
    def aggregate_neighbors(
        self,
        center_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_positions: torch.Tensor,
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """邻居特征聚合 - 不同的聚合策略"""
        if self.aggregation == 'mean':
            return self._aggregate_mean(neighbor_features, valid_mask)
        elif self.aggregation == 'max':
            return self._aggregate_max(neighbor_features, valid_mask)
        elif self.aggregation == 'weighted_mean':
            return self._aggregate_weighted_mean(neighbor_features, valid_mask)
        elif self.aggregation == 'attention':
            return self._aggregate_attention(center_features, neighbor_features, valid_mask)
        elif self.aggregation == 'gated':
            return self._aggregate_gated(center_features, neighbor_features, valid_mask)
        else:
            return self._aggregate_mean(neighbor_features, valid_mask)
    
    def _aggregate_mean(self, neighbor_features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """均值聚合"""
        valid_count = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        masked_features = torch.where(
            valid_mask.unsqueeze(-1),
            neighbor_features,
            torch.zeros_like(neighbor_features)
        )
        return masked_features.sum(dim=1) / valid_count
    
    def _aggregate_max(self, neighbor_features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """最大值聚合"""
        masked_features = torch.where(
            valid_mask.unsqueeze(-1),
            neighbor_features,
            torch.full_like(neighbor_features, -1e10)
        )
        return masked_features.max(dim=1)[0]
    
    def _aggregate_weighted_mean(self, neighbor_features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """加权均值聚合"""
        weights = self.weight_net(neighbor_features).squeeze(-1)  # (N, K)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        
        weight_sum = weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        weights = weights / weight_sum
        
        return (neighbor_features * weights.unsqueeze(-1)).sum(dim=1)
    
    def _aggregate_attention(
        self, 
        center_features: torch.Tensor, 
        neighbor_features: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """注意力聚合"""
        # 简化的注意力机制
        N, K, D = neighbor_features.shape
        
        query = self.query_proj(center_features).unsqueeze(1)  # (N, 1, D)
        key = self.key_proj(neighbor_features)                 # (N, K, D)
        value = self.value_proj(neighbor_features)             # (N, K, D)
        
        # 计算注意力权重
        attn_scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(D)  # (N, 1, K)
        attn_scores = attn_scores.squeeze(1)  # (N, K)
        
        # 应用有效性掩码
        attn_scores = torch.where(valid_mask, attn_scores, torch.full_like(attn_scores, -1e10))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 加权聚合
        return (value * attn_weights.unsqueeze(-1)).sum(dim=1)
    
    def _aggregate_gated(
        self, 
        center_features: torch.Tensor, 
        neighbor_features: torch.Tensor, 
        valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """门控聚合"""
        # 先做均值聚合
        neighbor_agg = self._aggregate_mean(neighbor_features, valid_mask)
        
        # 计算门控权重
        gate_input = torch.cat([center_features, neighbor_agg], dim=-1)
        gate = self.gate_net(gate_input)
        
        # 门控融合
        return gate * neighbor_agg + (1 - gate) * center_features
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        chunk_idx: Optional[torch.Tensor] = None,
        hist_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播"""
        B, N, _ = features.shape
        device = features.device
        
        feat_proj = self.feat_proj(features)
        enhanced_features = []
        
        for b in range(B):
            batch_features = feat_proj[b]
            batch_coords = coords[b]
            
            # 历史长度处理
            h_len = hist_len[b].item() if hist_len is not None else 0
            use_causal = self.causal and (chunk_idx is None or chunk_idx[b].item() > 0)
            
            # 找K近邻
            knn_idx, knn_dist, knn_valid = self.find_knn(
                batch_coords, self.k_neighbors, causal=use_causal
            )
            
            # 检查有效邻居
            has_neighbors = knn_valid.any(dim=1)
            enhanced = batch_features.clone()
            
            if has_neighbors.any():
                # 收集邻居特征
                neighbor_features = batch_features[knn_idx]  # (N, K, D)
                
                # 相对位置编码
                center_coords = batch_coords.unsqueeze(1)
                neighbor_coords = batch_coords[knn_idx]
                rel_pos = neighbor_coords - center_coords
                rel_dist = knn_dist.unsqueeze(-1)
                rel_encoding = torch.cat([rel_pos, rel_dist], dim=-1)
                pos_encoding = self.pos_encoder(rel_encoding)
                
                # 添加位置编码
                neighbor_features = neighbor_features + pos_encoding
                
                # 只对有邻居的点进行聚合
                points_with_neighbors = torch.where(has_neighbors)[0]
                if len(points_with_neighbors) > 0:
                    center_feat_subset = batch_features[points_with_neighbors]
                    neighbor_feat_subset = neighbor_features[points_with_neighbors]
                    valid_subset = knn_valid[points_with_neighbors]
                    
                    # 聚合邻居特征
                    agg_features = self.aggregate_neighbors(
                        center_feat_subset,
                        neighbor_feat_subset,
                        None,  # 位置信息已编码
                        valid_subset
                    )
                    
                    # 残差连接
                    enhanced[points_with_neighbors] = batch_features[points_with_neighbors] + \
                                                    self.dropout(agg_features)
            
            enhanced = self.norm(enhanced)
            enhanced_features.append(enhanced)
        
        return torch.stack(enhanced_features, dim=0)


def create_knn_variant(
    variant_type: str,
    input_dim: int,
    output_dim: int,
    **kwargs
) -> nn.Module:
    """创建KNN聚合变体"""
    
    # 预定义的变体配置
    variant_configs = {
        'mean': {'aggregation': 'mean'},
        'max': {'aggregation': 'max'},
        'weighted_mean': {'aggregation': 'weighted_mean'},
        'attention': {'aggregation': 'attention'},
        'gated': {'aggregation': 'gated'},
        'no_causal': {'causal': False},
        'large_radius': {'spatial_radius': 100.0},
        'small_k': {'k_neighbors': 8},
        'large_k': {'k_neighbors': 32}
    }
    
    if variant_type not in variant_configs:
        raise ValueError(f"Unknown KNN variant: {variant_type}")
    
    config = variant_configs[variant_type]
    config.update(kwargs)
    
    return KNNAggregationVariants(
        input_dim=input_dim,
        output_dim=output_dim,
        **config
    )