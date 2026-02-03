"""
空间编码器模块 - 支持因果KNN（修复版）
提供空间感知的特征编码方法
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# class SpatialKNNEncoder(nn.Module):
#     """
#     空间K近邻编码器，为每个事件点聚合局部空间上下文
#     支持因果KNN、半径过滤、无邻居回退
#     """
#     def __init__(
#         self,
#         input_dim: int,
#         output_dim: int,
#         k_neighbors: int = 16,
#         spatial_radius: Optional[float] = None,
#         aggregation: str = 'attention',
#         dropout: float = 0.1,
#         use_adaptive_temp: bool = True,
#         spatial_weight: float = 1.0,
#         temporal_weight: float = 0.3,
#         causal: bool = False  # 因果KNN开关
#     ):
#         super().__init__()
#         self.output_dim = output_dim
#         self.k_neighbors = k_neighbors
#         self.spatial_radius = spatial_radius
#         self.aggregation = aggregation
#         self.spatial_weight = spatial_weight
#         self.temporal_weight = temporal_weight
#         self.use_adaptive_temp = use_adaptive_temp
#         self.causal = causal  # 保存因果设置
        
#         # 特征投影
#         self.feat_proj = nn.Linear(input_dim, output_dim)
        
#         # 相对位置编码: [dx, dy, dt, dist] -> D
#         self.pos_encoder = nn.Sequential(
#             nn.Linear(4, output_dim // 2),
#             nn.ReLU(),
#             nn.Linear(output_dim // 2, output_dim)
#         )
        
#         # 聚合方式
#         if aggregation == 'attention':
#             self.attention = nn.MultiheadAttention(
#                 embed_dim=output_dim,
#                 num_heads=4,
#                 dropout=dropout,
#                 batch_first=True
#             )
#             self.query_proj = nn.Linear(output_dim, output_dim)
#             self.key_proj = nn.Linear(output_dim, output_dim)
#             self.value_proj = nn.Linear(output_dim, output_dim)
        
#         self.norm = nn.LayerNorm(output_dim)
#         self.dropout = nn.Dropout(dropout)

#         # 密度自适应温度参数
#         if use_adaptive_temp:
#             self.log_tau0 = nn.Parameter(torch.tensor(math.log(0.08)))
#             self.beta = nn.Parameter(torch.tensor(0.5))
    
#     def find_knn(self, coords: torch.Tensor, k: int, causal: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         找到每个点的K个最近邻
#         Args:
#             coords: (N, 3) [x, y, t]坐标
#             k: 邻居数量
#             causal: 是否使用因果屏蔽（屏蔽未来）
#         Returns:
#             indices: (N, k) 邻居索引
#             distances: (N, k) 距离
#             valid_mask: (N, k) 有效邻居掩码
#         """
#         N = coords.shape[0]
#         k = min(k, N)
        
#         # 计算成对距离矩阵
#         diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 3)
#         spatial_dist = torch.norm(diff[..., :2], dim=-1) * self.spatial_weight
#         temporal_dist = torch.abs(diff[..., 2]) * self.temporal_weight
#         distances = spatial_dist + temporal_dist
        
#         # 排除自循环
#         distances.fill_diagonal_(float('inf'))
        
#         # 因果屏蔽：屏蔽未来的点
#         if causal:
#             t = coords[:, 2]  # (N,)
#             future_mask = t.unsqueeze(0) > t.unsqueeze(1)  # (N, N), [i,j]=True表示j在i的未来
#             distances.masked_fill_(future_mask, float('inf'))
        
#         # 如果设置了空间半径，进行筛选
#         if self.spatial_radius is not None:
#             distances = torch.where(
#                 spatial_dist <= self.spatial_radius,
#                 distances,
#                 float('inf')
#             )
        
#         # 找K近邻
#         knn_dist, knn_idx = torch.topk(distances, k, dim=1, largest=False)
        
#         # 创建有效掩码（非inf的邻居）
#         valid_mask = torch.isfinite(knn_dist)
        
#         # 将inf距离替换为0（避免数值问题）
#         knn_dist = torch.where(valid_mask, knn_dist, torch.zeros_like(knn_dist))
        
#         return knn_idx, knn_dist, valid_mask
    
#     def compute_adaptive_temperature(self, knn_dist: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
#         """计算每个点的自适应温度"""
#         # 只考虑有效邻居的距离
#         masked_dist = torch.where(valid_mask, knn_dist, torch.zeros_like(knn_dist))
        
#         # 使用有效邻居的最大距离作为局部密度的估计
#         # 如果没有有效邻居，使用默认值
#         has_neighbors = valid_mask.any(dim=1)
#         rk = torch.zeros(knn_dist.shape[0], device=knn_dist.device)
        
#         if has_neighbors.any():
#             valid_max = masked_dist[has_neighbors].max(dim=1).values
#             rk[has_neighbors] = valid_max
        
#         rk = rk.detach() + 1e-6

#         # 相对化到batch中位数
#         if has_neighbors.any():
#             rk_rel = rk / (rk[has_neighbors].median() + 1e-6)
#         else:
#             rk_rel = torch.ones_like(rk)
            
#         tau_i = torch.exp(self.log_tau0) * (rk_rel ** self.beta)
#         tau_i = tau_i.clamp_(min=1e-3, max=1.0)
        
#         return tau_i
    
#     def forward(self, features: torch.Tensor, coords: torch.Tensor, 
#                 valid_mask: Optional[torch.Tensor] = None,
#                 chunk_idx: Optional[torch.Tensor] = None,
#                 hist_len: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         Args:
#             features: (B, N, input_dim) 事件特征
#             coords: (B, N, 3) 事件坐标 [x, y, t]
#             valid_mask: (B, N) 有效点掩码
#             chunk_idx: (B,) 每个批次的块序号
#             hist_len: (B,) 每个批次的历史长度
#         """
#         B, N, _ = features.shape
#         device = features.device
        
#         feat_proj = self.feat_proj(features)  # (B, N, D)
#         enhanced_features = []
        
#         for b in range(B):
#             batch_features = feat_proj[b]  # (N, D)
#             batch_coords = coords[b]       # (N, 3)
            
#             # 获取历史长度
#             h_len = hist_len[b].item() if hist_len is not None else 0
            
#             # 判断是否使用因果
#             # 第一个块（chunk_idx==0）或没有历史时不使用因果
#             use_causal = self.causal and (chunk_idx is None or chunk_idx[b].item() > 0)
            
#             # 找K近邻（全序列，包括历史）
#             knn_idx, knn_dist, knn_valid = self.find_knn(batch_coords, self.k_neighbors, causal=use_causal)
            
#             # 检查每个点是否有有效邻居
#             has_any_neighbor = knn_valid.any(dim=1)  # (N,)
            
#             # 初始化增强特征为原始特征
#             enhanced = batch_features.clone()
            
#             # 只对有邻居的点进行聚合
#             if has_any_neighbor.any():
#                 # 收集邻居特征
#                 neighbor_features = batch_features[knn_idx]  # (N, k, D)
                
#                 # 相对位置编码
#                 center_coords = batch_coords.unsqueeze(1)
#                 neighbor_coords = batch_coords[knn_idx]
#                 rel_pos = neighbor_coords - center_coords
#                 rel_dist = knn_dist.unsqueeze(-1)
#                 rel_encoding = torch.cat([rel_pos, rel_dist], dim=-1)
#                 pos_encoding = self.pos_encoder(rel_encoding)
                
#                 # 聚合邻居特征
#                 if self.aggregation == 'mean':
#                     # 使用有效掩码进行加权平均
#                     # 对于没有有效邻居的点，权重会是0
#                     valid_float = knn_valid.float()
#                     num_valid = valid_float.sum(dim=-1, keepdim=True).clamp(min=1)  # (N, 1)
                    
#                     # 计算有效邻居的权重
#                     masked_dist = torch.where(knn_valid, -knn_dist, torch.full_like(knn_dist, -1e10))
#                     weights = F.softmax(masked_dist, dim=-1).unsqueeze(-1)  # (N, k, 1)
                    
#                     # 加权聚合
#                     aggregated = (neighbor_features + pos_encoding) * weights
#                     aggregated = aggregated.sum(dim=1)  # (N, D)
                    
#                     # 只更新有邻居的点
#                     enhanced[has_any_neighbor] = batch_features[has_any_neighbor] + self.dropout(aggregated[has_any_neighbor])
                    
#                 elif self.aggregation == 'max':
#                     # 将无效邻居的特征设为-inf
#                     valid_features = neighbor_features + pos_encoding
#                     valid_features = torch.where(
#                         knn_valid.unsqueeze(-1),
#                         valid_features,
#                         torch.full_like(valid_features, -1e10)
#                     )
#                     aggregated = valid_features.max(dim=1)[0]
                    
#                     # 只更新有邻居的点
#                     enhanced[has_any_neighbor] = batch_features[has_any_neighbor] + self.dropout(aggregated[has_any_neighbor])
                
#                 elif self.aggregation == 'attention':
#                     # 对有邻居的点进行 attention
#                     points_with_neighbors = torch.where(has_any_neighbor)[0]
#                     if len(points_with_neighbors) > 0:
#                         M = points_with_neighbors.numel()
#                         klen = neighbor_features.shape[1]
#                         D = batch_features.shape[-1]
#                         H = self.attention.num_heads
#                         Dh = D // H

#                         # 1) q/k/v
#                         q = self.query_proj(batch_features[points_with_neighbors]).unsqueeze(1)  # (M,1,D)
#                         k = self.key_proj(
#                             neighbor_features[points_with_neighbors] + pos_encoding[points_with_neighbors]
#                         )  # (M,klen,D)
#                         v = self.value_proj(neighbor_features[points_with_neighbors])            # (M,klen,D)

#                         # 2) 计算自适应温度并形成距离先验偏置  prior = -d_ij / tau_i
#                         if getattr(self, "use_adaptive_temp", True):
#                             tau = self.compute_adaptive_temperature(knn_dist, knn_valid)  # (N,)
#                         else:
#                             # 退化为常数温度
#                             tau = torch.full((knn_dist.shape[0],), 0.08, device=knn_dist.device, dtype=knn_dist.dtype)
#                         tau_m = tau[points_with_neighbors].unsqueeze(-1)        # (M,1)

#                         prior = -knn_dist[points_with_neighbors] / (tau_m + 1e-6)        # (M,klen)
#                         # 无效邻居写成大负数，等价屏蔽
#                         prior = torch.where(
#                             knn_valid[points_with_neighbors],
#                             prior,
#                             torch.full_like(prior, -1e9)
#                         )  # (M,klen)

#                         # 3) 拆多头并调用 SDPA，attn_mask 作为“加性偏置”注入 logits
#                         #    形状变换：(M,*,D) -> (M*H,*,Dh)
#                         qh = q.view(M, 1, H, Dh).transpose(1, 2).reshape(M * H, 1, Dh)         # (M*H,1,Dh)
#                         kh = k.view(M, klen, H, Dh).transpose(1, 2).reshape(M * H, klen, Dh)   # (M*H,klen,Dh)
#                         vh = v.view(M, klen, H, Dh).transpose(1, 2).reshape(M * H, klen, Dh)   # (M*H,klen,Dh)

#                         # 将 prior 广播到各个头： (M,klen) -> (M*H,1,klen)
#                         attn_bias = prior.unsqueeze(1).repeat_interleave(H, dim=0)

#                         try:
#                             out = F.scaled_dot_product_attention(
#                                 qh, kh, vh, attn_mask=attn_bias, dropout_p=0.0, is_causal=False
#                             )  # (M*H,1,Dh)
#                         except TypeError:
#                             # 兼容老版本 PyTorch：手动计算
#                             scores = (qh @ kh.transpose(-2, -1)) / math.sqrt(Dh) + attn_bias   # (M*H,1,klen)
#                             attn = scores.softmax(dim=-1)
#                             out = attn @ vh                                                   # (M*H,1,Dh)

#                         # 拼回 (M,D)
#                         aggregated_subset = out.view(M, H, 1, Dh).transpose(1, 2).reshape(M, D)

#                         # 4) 残差 + dropout
#                         enhanced[points_with_neighbors] = batch_features[points_with_neighbors] + self.dropout(aggregated_subset)

                    
#                 # elif self.aggregation == 'attention':
#                 #     # 对有邻居的点进行attention
#                 #     points_with_neighbors = torch.where(has_any_neighbor)[0]
                    
#                 #     if len(points_with_neighbors) > 0:
#                 #         # 只处理有邻居的点
#                 #         q = self.query_proj(batch_features[points_with_neighbors]).unsqueeze(1)  # (M, 1, D)
#                 #         k = self.key_proj(neighbor_features[points_with_neighbors] + pos_encoding[points_with_neighbors])  # (M, k, D)
#                 #         v = self.value_proj(neighbor_features[points_with_neighbors])  # (M, k, D)
                        
#                 #         # 创建attention mask
#                 #         attn_mask = ~knn_valid[points_with_neighbors]  # (M, k)
#                 #         # 扩展mask以匹配attention的需求 (M, 1, k)
#                 #         attn_mask = attn_mask.unsqueeze(1)
                        
#                 #         # 应用attention（使用mask避免无效邻居）
#                 #         aggregated_subset, _ = self.attention(
#                 #             q, k, v,
#                 #             key_padding_mask=attn_mask.squeeze(1)
#                 #         )
#                 #         aggregated_subset = aggregated_subset.squeeze(1)  # (M, D)
                        
#                 #         # 更新有邻居的点
#                 #         enhanced[points_with_neighbors] = batch_features[points_with_neighbors] + self.dropout(aggregated_subset)
            
#             # 归一化（对所有点，包括没有邻居的）
#             enhanced = self.norm(enhanced)
#             enhanced_features.append(enhanced)
        
#         enhanced_features = torch.stack(enhanced_features, dim=0)
#         return enhanced_features


class SpatialKNNEncoder(nn.Module):
    """Simplified KNN encoder with only attention aggregation"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k_neighbors: int = 16,
        spatial_radius: Optional[float] = 50.0,
        aggregation: str = 'attention',
        causal: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.spatial_radius = spatial_radius
        self.causal = causal
        
        # Feature projection
        self.feat_proj = nn.Linear(input_dim, output_dim)
        
        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        # Attention aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def find_knn(self, coords: torch.Tensor, k: int, causal: bool = False):
        """Find K nearest neighbors with optional causal masking"""
        N = coords.shape[0]
        k = min(k, N)
        
        # Compute pairwise distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        spatial_dist = torch.norm(diff[..., :2], dim=-1)
        temporal_dist = torch.abs(diff[..., 2]) * 0.3
        distances = spatial_dist + temporal_dist
        
        # Self-loop exclusion
        distances.fill_diagonal_(float('inf'))
        
        # Causal masking (only look at past)
        if causal:
            t = coords[:, 2]
            future_mask = t.unsqueeze(0) > t.unsqueeze(1)
            distances.masked_fill_(future_mask, float('inf'))
        
        # Spatial radius filtering
        if self.spatial_radius is not None:
            distances = torch.where(
                spatial_dist <= self.spatial_radius,
                distances,
                float('inf')
            )
        
        # Find K neighbors
        knn_dist, knn_idx = torch.topk(distances, k, dim=1, largest=False)
        valid_mask = torch.isfinite(knn_dist)
        knn_dist = torch.where(valid_mask, knn_dist, torch.zeros_like(knn_dist))
        
        return knn_idx, knn_dist, valid_mask
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        chunk_idx: Optional[torch.Tensor] = None,
        hist_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with attention aggregation"""
        B, N, _ = features.shape
        
        feat_proj = self.feat_proj(features)
        enhanced_features = []
        
        for b in range(B):
            batch_features = feat_proj[b]
            batch_coords = coords[b]
            
            # Determine if we should use causal masking
            use_causal = self.causal and (chunk_idx is None or chunk_idx[b].item() > 0)
            
            # Find KNN
            knn_idx, knn_dist, knn_valid = self.find_knn(
                batch_coords, self.k_neighbors, causal=use_causal
            )
            
            # Check for valid neighbors
            has_neighbors = knn_valid.any(dim=1)
            enhanced = batch_features.clone()
            
            if has_neighbors.any():
                # Collect neighbor features
                neighbor_features = batch_features[knn_idx]
                
                # Relative position encoding
                center_coords = batch_coords.unsqueeze(1)
                neighbor_coords = batch_coords[knn_idx]
                rel_pos = neighbor_coords - center_coords
                rel_dist = knn_dist.unsqueeze(-1)
                rel_encoding = torch.cat([rel_pos, rel_dist], dim=-1)
                pos_encoding = self.pos_encoder(rel_encoding)
                
                # Add position encoding
                neighbor_features = neighbor_features + pos_encoding
                
                # Process points with neighbors
                points_with_neighbors = torch.where(has_neighbors)[0]
                if len(points_with_neighbors) > 0:
                    # Attention aggregation
                    q = batch_features[points_with_neighbors].unsqueeze(1)
                    k = neighbor_features[points_with_neighbors]
                    v = neighbor_features[points_with_neighbors]
                    
                    # Create attention mask for invalid neighbors
                    attn_mask = ~knn_valid[points_with_neighbors]
                    
                    # Apply attention
                    aggregated, _ = self.attention(
                        q, k, v,
                        key_padding_mask=attn_mask
                    )
                    aggregated = aggregated.squeeze(1)
                    
                    # Residual connection
                    enhanced[points_with_neighbors] = (
                        batch_features[points_with_neighbors] + 
                        self.dropout(aggregated)
                    )
            
            enhanced = self.norm(enhanced)
            enhanced_features.append(enhanced)
        
        return torch.stack(enhanced_features, dim=0)