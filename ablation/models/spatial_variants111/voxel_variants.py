"""
Micro-voxel spatial encoder - Fixed version
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class MicroVoxelSpatialEncoder(nn.Module):
    """
    Voxel-based spatial encoder for 3D (x,y,t) space
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        voxel_size: Tuple[float, float, float] = (2.0, 2.0, 0.005),
        neighborhood_size: int = 1,
        aggregation: str = 'attention',
        dropout: float = 0.1,
        space_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        max_points_per_voxel: int = 32
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.voxel_size = voxel_size
        self.neighborhood_size = neighborhood_size
        self.aggregation = aggregation
        self.max_points_per_voxel = max_points_per_voxel
        
        # Default space bounds
        if space_bounds is None:
            space_bounds = {
                'x': (0, 256),
                'y': (0, 256), 
                't': (0, 1.0)
            }
        self.space_bounds = space_bounds
        
        # Calculate grid dimensions
        self.grid_dims = []
        for i, dim in enumerate(['x', 'y', 't']):
            min_val, max_val = space_bounds[dim]
            n_voxels = max(1, int((max_val - min_val) / voxel_size[i]))
            self.grid_dims.append(n_voxels)
        
        # Feature projection
        self.feat_proj = nn.Linear(input_dim, output_dim)
        
        # Intra-voxel aggregation
        self.intra_voxel_agg = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )
        
        # Inter-voxel attention
        self.inter_voxel_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def _coords_to_voxel_idx(self, coords: torch.Tensor) -> torch.Tensor:
        """Map coordinates to voxel indices"""
        voxel_idx = torch.zeros_like(coords, dtype=torch.long)
        
        for i, dim in enumerate(['x', 'y', 't']):
            min_val, max_val = self.space_bounds[dim]
            # Normalize and quantize to voxel grid
            norm_coord = (coords[:, i] - min_val) / (max_val - min_val)
            norm_coord = torch.clamp(norm_coord, 0, 1)
            voxel_idx[:, i] = (norm_coord * (self.grid_dims[i] - 1)).long()
        
        return voxel_idx
    
    def _get_neighbor_offsets(self) -> torch.Tensor:
        """Generate 3x3x3 neighborhood offsets"""
        offsets = []
        n = self.neighborhood_size
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                for dt in range(-n, n + 1):
                    offsets.append([dx, dy, dt])
        return torch.tensor(offsets, dtype=torch.long)
    
    def _aggregate_voxel_features(
        self, 
        point_features: torch.Tensor,
        point_voxel_idx: torch.Tensor,
        unique_voxels: torch.Tensor
    ) -> Dict[Tuple, torch.Tensor]:
        """Aggregate features within each voxel"""
        voxel_features = {}
        
        for voxel in unique_voxels:
            # Find all points belonging to this voxel
            mask = (point_voxel_idx == voxel.unsqueeze(0)).all(dim=-1)
            if not mask.any():
                continue
            
            voxel_points = point_features[mask]
            
            # Limit max points per voxel
            if len(voxel_points) > self.max_points_per_voxel:
                indices = torch.randperm(len(voxel_points))[:self.max_points_per_voxel]
                voxel_points = voxel_points[indices]
            
            # Intra-voxel aggregation
            if len(voxel_points) == 1:
                agg_feat = voxel_points[0]
            else:
                agg_feat = voxel_points.mean(dim=0)
            
            # Further processing
            agg_feat = self.intra_voxel_agg(agg_feat)
            voxel_key = tuple(voxel.tolist())
            voxel_features[voxel_key] = agg_feat
        
        return voxel_features
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        chunk_idx: Optional[torch.Tensor] = None,
        hist_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        B, N, _ = features.shape
        device = features.device
        
        feat_proj = self.feat_proj(features)
        enhanced_features = []
        
        # Pre-compute neighborhood offsets
        neighbor_offsets = self._get_neighbor_offsets().to(device)
        
        for b in range(B):
            batch_features = feat_proj[b]
            batch_coords = coords[b]
            
            # Map to voxel indices
            voxel_idx = self._coords_to_voxel_idx(batch_coords)
            
            # Find unique voxels
            unique_voxels = torch.unique(voxel_idx, dim=0)
            
            # Aggregate features within voxels
            voxel_features = self._aggregate_voxel_features(
                batch_features, voxel_idx, unique_voxels
            )
            
            # Enhance each point
            enhanced = batch_features.clone()
            
            for i in range(N):
                point_voxel = tuple(voxel_idx[i].tolist())
                
                # Collect neighborhood voxel features
                neighbor_feats = []
                neighbor_positions = []
                
                for offset in neighbor_offsets:
                    neighbor_voxel_idx = voxel_idx[i] + offset
                    
                    # Boundary check
                    valid = True
                    for dim in range(3):
                        if (neighbor_voxel_idx[dim] < 0 or 
                            neighbor_voxel_idx[dim] >= self.grid_dims[dim]):
                            valid = False
                            break
                    
                    if valid:
                        neighbor_key = tuple(neighbor_voxel_idx.tolist())
                        if neighbor_key in voxel_features:
                            neighbor_feats.append(voxel_features[neighbor_key])
                            neighbor_positions.append(offset.float())
                
                if len(neighbor_feats) > 0:
                    neighbor_feats = torch.stack(neighbor_feats, dim=0)
                    neighbor_positions = torch.stack(neighbor_positions, dim=0)
                    
                    # Position encoding
                    pos_encoding = self.pos_encoder(neighbor_positions)
                    neighbor_feats = neighbor_feats + pos_encoding
                    
                    # Attention aggregation
                    query = batch_features[i:i+1].unsqueeze(0)
                    key_val = neighbor_feats.unsqueeze(0)
                    
                    attn_out, _ = self.inter_voxel_attention(query, key_val, key_val)
                    agg_feat = attn_out.squeeze(0).squeeze(0)
                    
                    # Residual connection
                    enhanced[i] = batch_features[i] + self.dropout(agg_feat)
            
            enhanced = self.norm(enhanced)
            enhanced_features.append(enhanced)
        
        return torch.stack(enhanced_features, dim=0)