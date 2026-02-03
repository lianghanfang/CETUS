"""
Grid-based spatial encoder - Fixed version
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GridSpatialEncoder(nn.Module):
    """Grid-based spatial encoder with attention aggregation"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: Tuple[int, int] = (8, 8),
        neighborhood: int = 1,
        aggregation: str = 'attention',
        dropout: float = 0.1,
        image_size: Tuple[int, int] = (256, 256)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.neighborhood = neighborhood
        self.aggregation = aggregation
        self.image_size = image_size
        
        # Cell dimensions
        self.cell_width = image_size[0] / grid_size[0]
        self.cell_height = image_size[1] / grid_size[1]
        
        # Feature projection
        self.feat_proj = nn.Linear(input_dim, output_dim)
        
        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, output_dim // 2),
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
    
    def _coords_to_grid_idx(self, coords: torch.Tensor) -> torch.Tensor:
        """Map coordinates to grid indices"""
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Map to grid indices
        grid_x = torch.clamp((x / self.cell_width).long(), 0, self.grid_size[0] - 1)
        grid_y = torch.clamp((y / self.cell_height).long(), 0, self.grid_size[1] - 1)
        
        return torch.stack([grid_x, grid_y], dim=1)
    
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
        
        feat_proj = self.feat_proj(features)
        enhanced_features = []
        
        for b in range(B):
            batch_features = feat_proj[b]
            batch_coords = coords[b]
            
            # Map to grid
            grid_indices = self._coords_to_grid_idx(batch_coords)
            
            # Create grid feature map
            grid_features = {}
            grid_counts = {}
            
            # Aggregate features per grid cell
            for i in range(N):
                gx, gy = grid_indices[i].tolist()
                cell_key = (gx, gy)
                
                if cell_key not in grid_features:
                    grid_features[cell_key] = batch_features[i].clone()
                    grid_counts[cell_key] = 1
                else:
                    grid_features[cell_key] = grid_features[cell_key] + batch_features[i]
                    grid_counts[cell_key] = grid_counts[cell_key] + 1
            
            # Normalize grid features
            for cell_key in grid_features:
                grid_features[cell_key] = grid_features[cell_key] / grid_counts[cell_key]
            
            # Enhance each point
            enhanced = batch_features.clone()
            
            for i in range(N):
                gx, gy = grid_indices[i].tolist()
                
                # Collect neighborhood features
                neighbor_feats = []
                neighbor_positions = []
                
                for dx in range(-self.neighborhood, self.neighborhood + 1):
                    for dy in range(-self.neighborhood, self.neighborhood + 1):
                        nx, ny = gx + dx, gy + dy
                        
                        if (0 <= nx < self.grid_size[0] and 
                            0 <= ny < self.grid_size[1] and 
                            (nx, ny) in grid_features):
                            
                            neighbor_feats.append(grid_features[(nx, ny)])
                            neighbor_positions.append(
                                torch.tensor([dx, dy], dtype=torch.float, device=features.device)
                            )
                
                if len(neighbor_feats) > 0:
                    neighbor_feats = torch.stack(neighbor_feats, dim=0)
                    neighbor_positions = torch.stack(neighbor_positions, dim=0)
                    
                    # Position encoding
                    pos_encoding = self.pos_encoder(neighbor_positions)
                    neighbor_feats = neighbor_feats + pos_encoding
                    
                    # Attention aggregation
                    query = batch_features[i:i+1].unsqueeze(0)
                    key_val = neighbor_feats.unsqueeze(0)
                    
                    attn_out, _ = self.attention(query, key_val, key_val)
                    agg_feat = attn_out.squeeze(0).squeeze(0)
                    
                    # Residual connection
                    enhanced[i] = batch_features[i] + self.dropout(agg_feat)
            
            enhanced = self.norm(enhanced)
            enhanced_features.append(enhanced)
        
        return torch.stack(enhanced_features, dim=0)