"""
Spatial encoder factory and variants for ablation experiments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _build_offsets_2d(n: int, device: torch.device) -> torch.Tensor:
    """Return (K, 2) integer offsets for a square neighborhood radius n.
    Order is row-major from (-n,-n) to (n,n)."""
    xs = torch.arange(-n, n + 1, device=device)
    ys = torch.arange(-n, n + 1, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (Kside,Kside)
    offsets = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (K,2)
    return offsets.to(torch.long)


def _build_offsets_3d(n: int, device: torch.device) -> torch.Tensor:
    """Return (K, 3) integer offsets for a cubic neighborhood radius n."""
    xs = torch.arange(-n, n + 1, device=device)
    ys = torch.arange(-n, n + 1, device=device)
    zs = torch.arange(-n, n + 1, device=device)
    gz, gy, gx = torch.meshgrid(zs, ys, xs, indexing="ij")
    offsets = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    return offsets.to(torch.long)


def _index_add_mean(
    feats: torch.Tensor,  # (B,N,D)
    idx: torch.Tensor,    # (B,N) in [0, M)
    M: int,
) -> torch.Tensor:
    """Mean pooling by index using index_add_ (faster & less contention than scatter_add_).
    Returns (B,M,D) where empty bins are zero (mean over 0 -> 0)."""
    B, N, D = feats.shape
    dev = feats.device
    # Flatten (B,N) -> (B*N) global indices
    base = (torch.arange(B, device=dev) * M).unsqueeze(1)  # (B,1)
    gid = (idx + base).reshape(-1)  # (B*N,)
    # Accumulators
    out = torch.zeros(B * M, D, device=dev, dtype=feats.dtype)
    cnt = torch.zeros(B * M, device=dev, dtype=feats.dtype)
    out.index_add_(0, gid, feats.reshape(-1, D))
    cnt.index_add_(0, gid, torch.ones_like(gid, dtype=feats.dtype))
    cnt = cnt.clamp_min_(1.0).unsqueeze(-1)  # avoid div by zero
    out = (out / cnt).view(B, M, D)
    return out


def _gather_linear(
    table: torch.Tensor,      # (B,M,D)
    linear_idx: torch.Tensor, # (B,*,K) in [0,M)
) -> torch.Tensor:
    """Vectorized gather by linear indices without huge expands.
    Returns (B,*,K,D)."""
    B, M, D = table.shape
    dev = table.device
    flat = table.reshape(B * M, D)  # (B*M, D)
    base = (torch.arange(B, device=dev) * M).view(B, 1, 1)
    sel = (linear_idx + base).reshape(-1)  # (B* * K,)
    out = flat.index_select(0, sel)  # (B* * K, D)
    return out.view(*linear_idx.shape, D)


# -----------------------------------------------------------------------------
# Grid-based Spatial Encoder (2D)
# -----------------------------------------------------------------------------

class GridSpatialEncoder(nn.Module):
    """Grid-based spatial encoder with efficient batched attention.

    Interface:
        __init__(input_dim, output_dim, grid_size=(8,8), neighborhood=1,
                 aggregation='attention', dropout=0.1, image_size=(W,H),
                 num_heads=1, use_pos_embedding=True)
        forward(features: (B,N,Fin), coords: (B,N,2), valid_mask: Optional[(B,N)])
            -> (B,N,Fout)

    Notes:
    - Neighborhood of radius n yields K=(2n+1)^2 neighbors per point (with edge clamping).
    - Uses F.scaled_dot_product_attention for speed; falls back to manual path if needed.
    - "mean" aggregation available for very memory-lean setups.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_size: Tuple[int, int] = (8, 8),
        neighborhood: int = 1,
        aggregation: str = 'attention',
        dropout: float = 0.1,
        image_size: Tuple[int, int] = (256, 256),
        num_heads: int = 1,
        use_pos_embedding: bool = True,
    ):
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.neighborhood = int(neighborhood)
        self.aggregation = aggregation
        self.image_size = image_size
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_pos_embedding = use_pos_embedding

        # Cell dimensions
        self.cell_w = float(image_size[0]) / float(grid_size[0])
        self.cell_h = float(image_size[1]) / float(grid_size[1])

        # Projections
        self.feat_proj = nn.Linear(input_dim, output_dim)
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

        # Positional embedding for relative (dx,dy)
        K = (2 * self.neighborhood + 1) ** 2
        if use_pos_embedding:
            self.pos_emb = nn.Embedding(K, output_dim)
        else:
            self.pos_emb = None

        # Precompute offsets and ids as buffers
        offsets = _build_offsets_2d(self.neighborhood, device=torch.device('cpu'))  # moved to cuda lazily
        self.register_buffer('offsets_2d', offsets, persistent=False)
        self.register_buffer('pos_ids_2d', torch.arange(K, dtype=torch.long), persistent=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    @torch.no_grad()
    def _coords_to_grid_idx(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B,N,2) in pixels; map to (B,N,2) integer cell indices
        x = coords[..., 0]
        y = coords[..., 1]
        gx, gy = self.grid_size
        ix = torch.clamp((x / self.cell_w).long(), 0, gx - 1)
        iy = torch.clamp((y / self.cell_h).long(), 0, gy - 1)
        return torch.stack([ix, iy], dim=-1)  # (B,N,2)

    def _build_grid_features(self, feat: torch.Tensor, grid_idx: torch.Tensor) -> torch.Tensor:
        # Mean feature per cell: (B,N,D) -> (B,gx,gy,D)
        gx, gy = self.grid_size
        M = gx * gy
        lin = grid_idx[..., 0] * gy + grid_idx[..., 1]  # (B,N)
        pooled = _index_add_mean(feat, lin, M)  # (B,M,D)
        return pooled.view(feat.shape[0], gx, gy, -1)

    def _gather_neighbors(self, grid_feat: torch.Tensor, grid_idx: torch.Tensor):
        # Gather K neighbors for every point; returns (B,N,K,D) and valid mask (B,N,K)
        B, N, _ = grid_idx.shape
        gx, gy = self.grid_size
        dev = grid_feat.device

        # (K,2) -> (1,1,K,2)
        ofs = self.offsets_2d.to(dev).view(1, 1, -1, 2)

        # Center indices (B,N,1,2) + offsets -> (B,N,K,2)
        nbr = grid_idx.unsqueeze(2) + ofs

        # Validity before clamping
        valid = (
            (nbr[..., 0] >= 0) & (nbr[..., 0] < gx) &
            (nbr[..., 1] >= 0) & (nbr[..., 1] < gy)
        )  # (B,N,K)

        # Clamp for safe gather
        nbr_clamped = torch.stack([
            nbr[..., 0].clamp(0, gx - 1),
            nbr[..., 1].clamp(0, gy - 1)
        ], dim=-1)

        # Linearize & gather without huge expands
        M = gx * gy
        lin = nbr_clamped[..., 0] * gy + nbr_clamped[..., 1]  # (B,N,K)
        grid_flat = grid_feat.view(B, M, self.output_dim)     # (B,M,D)
        nbr_feat = _gather_linear(grid_flat, lin)             # (B,N,K,D)

        # Add positional embedding
        if self.pos_emb is not None:
            pos = self.pos_emb(self.pos_ids_2d.to(dev)).view(1, 1, -1, self.output_dim)
            nbr_feat = nbr_feat + pos

        return nbr_feat, valid

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        chunk_idx: Optional[torch.Tensor] = None,   # 兼容位置参数4
        hist_len: Optional[torch.Tensor] = None,    # 兼容位置参数5
        **kwargs
    ) -> torch.Tensor:
        B, N, _ = features.shape
        dev = features.device

        feat = self.feat_proj(features)  # (B,N,D)
        grid_idx = self._coords_to_grid_idx(coords[..., :2])  # (B,N,2)
        grid_feat = self._build_grid_features(feat, grid_idx)  # (B,gx,gy,D)
        nbr_feat, valid = self._gather_neighbors(grid_feat, grid_idx)  # (B,N,K,D), (B,N,K)

        if self.aggregation == 'mean':
            # Uniform mean over valid neighbors
            w = valid.float()
            w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1.0))  # (B,N,K)
            out = (nbr_feat * w.unsqueeze(-1)).sum(dim=2)  # (B,N,D)
        else:
            # Attention via fused SDPA (batched over BN queries)
            Q = self.q_proj(feat).unsqueeze(2)           # (B,N,1,D)
            K = self.k_proj(nbr_feat)                    # (B,N,K,D)
            V = self.v_proj(nbr_feat)                    # (B,N,K,D)

            # Merge BN and call SDPA once
            BN, Klen, D = B * N, K.shape[2], self.output_dim
            Qh = Q.reshape(BN, 1, D)
            Kh = K.reshape(BN, Klen, D)
            Vh = V.reshape(BN, Klen, D)

            # Build additive mask: invalid -> -inf
            bias = torch.zeros(BN, 1, Klen, device=dev, dtype=Qh.dtype)
            if not valid.all():
                bias.masked_fill_(~valid.reshape(BN, Klen).unsqueeze(1), float('-inf'))

            try:
                out = F.scaled_dot_product_attention(Qh, Kh, Vh, attn_mask=bias, dropout_p=0.0)
            except TypeError:
                # Fallback: manual path (older PyTorch)
                scores = (Qh @ Kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
                scores = scores + bias
                attn = torch.softmax(scores, dim=-1)
                out = attn @ Vh

            out = out.reshape(B, N, 1, D).squeeze(2)  # (B,N,D)
            out = self.out_proj(out)

        out = feat + self.dropout(out)
        out = self.norm(out)

        if valid_mask is not None:
            out = torch.where(valid_mask.unsqueeze(-1), out, feat)
        return out


# -----------------------------------------------------------------------------
# Voxel-based Spatial Encoder (3D: x,y,t)
# -----------------------------------------------------------------------------

class MicroVoxelSpatialEncoder(nn.Module):
    """Voxel-based spatial encoder with efficient batched attention.

    Interface:
        __init__(input_dim, output_dim, voxel_size=(sx,sy,st), neighborhood_size=1,
                 aggregation='attention', dropout=0.1,
                 space_bounds={'x':(0,W), 'y':(0,H), 't':(t0,t1)},
                 num_heads=1, use_pos_embedding=True)
        forward(features: (B,N,Fin), coords: (B,N,3), valid_mask: Optional[(B,N)])
            -> (B,N,Fout)

    Implementation details:
    - Per-voxel mean features computed via index_add_ (no Python dict / loops).
    - Neighbor gather uses linearized voxel index + index_select (no huge expands).
    - SDPA for fast attention; fallback to manual path if needed.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.005),
        neighborhood_size: int = 1,
        aggregation: str = 'attention',
        dropout: float = 0.1,
        space_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        num_heads: int = 1,
        use_pos_embedding: bool = True,
    ):
        super().__init__()
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.voxel_size = voxel_size
        self.nh = int(neighborhood_size)
        self.aggregation = aggregation
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.use_pos_embedding = use_pos_embedding

        if space_bounds is None:
            space_bounds = {'x': (0.0, 1.0), 'y': (0.0, 1.0), 't': (0.0, 1.0)}
        self.space_bounds = space_bounds

        # Derive voxel grid dims
        self.grid_dims = self._compute_grid_dims()
        Nx, Ny, Nt = self.grid_dims
        self.M = Nx * Ny * Nt  # total voxels

        # Projections
        self.feat_proj = nn.Linear(input_dim, output_dim)
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

        # Pos embedding over (dx,dy,dt)
        K = (2 * self.nh + 1) ** 3
        if use_pos_embedding:
            self.pos_emb = nn.Embedding(K, output_dim)
        else:
            self.pos_emb = None

        offsets3d = _build_offsets_3d(self.nh, device=torch.device('cpu'))
        self.register_buffer('offsets_3d', offsets3d, persistent=False)
        self.register_buffer('pos_ids_3d', torch.arange(K, dtype=torch.long), persistent=False)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)

    def _compute_grid_dims(self):
        Nx = max(1, int((self.space_bounds['x'][1] - self.space_bounds['x'][0]) / float(self.voxel_size[0])))
        Ny = max(1, int((self.space_bounds['y'][1] - self.space_bounds['y'][0]) / float(self.voxel_size[1])))
        Nt = max(1, int((self.space_bounds['t'][1] - self.space_bounds['t'][0]) / float(self.voxel_size[2])))
        return (Nx, Ny, Nt)

    @torch.no_grad()
    def _coords_to_voxel_idx(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B,N,3) in absolute units matching space_bounds
        x = coords[..., 0]; y = coords[..., 1]; t = coords[..., 2]
        (x0, x1), (y0, y1), (t0, t1) = self.space_bounds['x'], self.space_bounds['y'], self.space_bounds['t']
        Nx, Ny, Nt = self.grid_dims

        # normalize to [0,1] then quantize to [0,Nx-1] etc.
        ix = ((x - x0) / (x1 - x0)).clamp(0, 1) * (Nx - 1)
        iy = ((y - y0) / (y1 - y0)).clamp(0, 1) * (Ny - 1)
        it = ((t - t0) / (t1 - t0)).clamp(0, 1) * (Nt - 1)
        return torch.stack([ix.long(), iy.long(), it.long()], dim=-1)  # (B,N,3)

    def _linearize(self, vidx: torch.Tensor) -> torch.Tensor:
        Nx, Ny, Nt = self.grid_dims
        return vidx[..., 0] + vidx[..., 1] * Nx + vidx[..., 2] * (Nx * Ny)

    def _build_voxel_features(self, feat: torch.Tensor, vidx: torch.Tensor) -> torch.Tensor:
        # Mean feature per voxel: (B,N,D) -> (B,Nx,Ny,Nt,D)
        B, N, D = feat.shape
        Nx, Ny, Nt = self.grid_dims
        M = self.M
        lin = self._linearize(vidx)  # (B,N)
        pooled = _index_add_mean(feat, lin, M)  # (B,M,D)
        return pooled.view(B, Nx, Ny, Nt, D)

    def _gather_neighbors(self, vox_feat: torch.Tensor, vidx: torch.Tensor):
        # Returns (B,N,K,D) and valid mask (B,N,K)
        B, N, _ = vidx.shape
        Nx, Ny, Nt = self.grid_dims
        dev = vox_feat.device

        ofs = self.offsets_3d.to(dev).view(1, 1, -1, 3)  # (1,1,K,3)
        nbr = vidx.unsqueeze(2) + ofs                    # (B,N,K,3)

        valid = (
            (nbr[..., 0] >= 0) & (nbr[..., 0] < Nx) &
            (nbr[..., 1] >= 0) & (nbr[..., 1] < Ny) &
            (nbr[..., 2] >= 0) & (nbr[..., 2] < Nt)
        )

        nbr_clamped = torch.stack([
            nbr[..., 0].clamp(0, Nx - 1),
            nbr[..., 1].clamp(0, Ny - 1),
            nbr[..., 2].clamp(0, Nt - 1),
        ], dim=-1)

        lin = self._linearize(nbr_clamped)  # (B,N,K)
        table = vox_feat.view(B, self.M, self.output_dim)  # (B,M,D)
        nbr_feat = _gather_linear(table, lin)              # (B,N,K,D)

        if self.pos_emb is not None:
            pos = self.pos_emb(self.pos_ids_3d.to(dev)).view(1, 1, -1, self.output_dim)
            nbr_feat = nbr_feat + pos

        return nbr_feat, valid

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        chunk_idx: Optional[torch.Tensor] = None,   # 兼容位置参数4
        hist_len: Optional[torch.Tensor] = None,    # 兼容位置参数5
        **kwargs
    ) -> torch.Tensor:
        B, N, _ = features.shape
        dev = features.device

        feat = self.feat_proj(features)              # (B,N,D)
        vidx = self._coords_to_voxel_idx(coords)     # (B,N,3)
        vox_feat = self._build_voxel_features(feat, vidx)  # (B,Nx,Ny,Nt,D)
        nbr_feat, valid = self._gather_neighbors(vox_feat, vidx)  # (B,N,K,D), (B,N,K)

        if self.aggregation == 'mean':
            w = valid.float()
            w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1.0))
            out = (nbr_feat * w.unsqueeze(-1)).sum(dim=2)  # (B,N,D)
        else:
            Q = self.q_proj(feat).unsqueeze(2)          # (B,N,1,D)
            K = self.k_proj(nbr_feat)                   # (B,N,K,D)
            V = self.v_proj(nbr_feat)                   # (B,N,K,D)

            BN, Klen, D = B * N, K.shape[2], self.output_dim
            Qh = Q.reshape(BN, 1, D)
            Kh = K.reshape(BN, Klen, D)
            Vh = V.reshape(BN, Klen, D)

            bias = torch.zeros(BN, 1, Klen, device=dev, dtype=Qh.dtype)
            if not valid.all():
                bias.masked_fill_(~valid.reshape(BN, Klen).unsqueeze(1), float('-inf'))

            try:
                out = F.scaled_dot_product_attention(Qh, Kh, Vh, attn_mask=bias, dropout_p=0.0)
            except TypeError:
                scores = (Qh @ Kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
                scores = scores + bias
                attn = torch.softmax(scores, dim=-1)
                out = attn @ Vh

            out = out.reshape(B, N, 1, D).squeeze(2)  # (B,N,D)
            out = self.out_proj(out)

        out = feat + self.dropout(out)
        out = self.norm(out)
        if valid_mask is not None:
            out = torch.where(valid_mask.unsqueeze(-1), out, feat)
        return out


#############################################################
#   knn spatial encoder
#############################################################

class BaseKNNSpatialEncoder(nn.Module):
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


class KNNSpatialEncoder(nn.Module):
    """
    KNN encoder with ring-based multi-scale aggregation + soft gating (可开关)
    - 一次取 K_total 个邻居，按距离切成若干环（near/mid/far）
    - 每个环独立做一次聚合（沿用同一个 MHA，参数共享，稳）
    - 用一个小门控 MLP 根据局部密度/时间结构生成 ring 权重（softmax）
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        k_neighbors: int = 16,
        spatial_radius: Optional[float] = 50.0,
        aggregation: str = 'attention',
        causal: bool = True,
        dropout: float = 0.1,
        # === 新增：环域多尺度 ===
        use_rings: bool = True,
        k_total: Optional[int] = None,          # 总邻居数（若为 None，默认= k_neighbors*3）
        ring_sizes: Optional[list[int]] = None, # 例如 [16,16,16]；长度= num_rings
        gate_use_time: bool = True,             # 门控是否使用时间结构特征
        gate_hidden: int = 32
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.spatial_radius = spatial_radius
        self.causal = causal
        self.use_rings = use_rings

        # --- ring 配置 ---
        if self.use_rings:
            self.k_total = int(k_total) if k_total is not None else int(k_neighbors * 3)
            if ring_sizes is None:
                # 均分到 3 个环
                base = self.k_total // 3
                self.ring_sizes = [base, base, self.k_total - 2*base]
            else:
                assert sum(ring_sizes) == self.k_total and len(ring_sizes) >= 2
                self.ring_sizes = ring_sizes
            self.num_rings = len(self.ring_sizes)
        else:
            self.k_total = k_neighbors
            self.ring_sizes = [k_neighbors]
            self.num_rings = 1

        # Feature projection
        self.feat_proj = nn.Linear(input_dim, output_dim)

        # Position encoding: [dx, dy, dt, dist] -> D
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim)
        )

        # Attention aggregation（共享一套参数，稳定）
        assert aggregation == 'attention', "当前实现固定使用 attention 聚合"
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Ring gate（密度/时间 软门控）
        gate_in_dim = 2  # [dens_proxy, valid_ratio]
        if gate_use_time:
            gate_in_dim += 2  # [+ mean|dt|, std|dt|]
        self.gate_use_time = gate_use_time

        self.ring_gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, self.num_rings)  # 输出每个 ring 的 raw logits
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _safe_norm(x: torch.Tensor, eps: float = 1e-6):
        # 简单鲁棒归一化：除以(中位数+eps)，再 clamp
        denom = x.median(dim=-1, keepdim=True).values + eps
        return torch.clamp(x / denom, 0.0, 10.0)

    def find_knn(self, coords: torch.Tensor, k: int, causal: bool = False):
        """
        Find K nearest neighbors with optional causal masking
        coords: (N, 3) -> (x, y, t)
        return: knn_idx (N,k), knn_dist (N,k), knn_valid (N,k)
        """
        N = coords.shape[0]
        k = min(k, N)

        # pairwise
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N,N,3)
        spatial_dist = torch.norm(diff[..., :2], dim=-1)  # (N,N)
        temporal_dist = torch.abs(diff[..., 2]) * 0.3     # 权重可调
        distances = spatial_dist + temporal_dist          # (N,N)

        # 自环无效
        distances.fill_diagonal_(float('inf'))

        # 因果遮罩：只看过去（t_j <= t_i）
        if causal:
            t = coords[:, 2]
            future_mask = t.unsqueeze(0) > t.unsqueeze(1)   # (N,N) True 表示 j 在未来
            distances.masked_fill_(future_mask, float('inf'))

        # 半径过滤（只在空间上过滤；时间已混入距离）
        if self.spatial_radius is not None:
            distances = torch.where(
                spatial_dist <= self.spatial_radius,
                distances,
                float('inf')
            )

        # top-k
        knn_dist, knn_idx = torch.topk(distances, k, dim=1, largest=False)
        knn_valid = torch.isfinite(knn_dist)
        knn_dist = torch.where(knn_valid, knn_dist, torch.zeros_like(knn_dist))
        return knn_idx, knn_dist, knn_valid

    def _split_rings(self, knn_idx: torch.Tensor, knn_dist: torch.Tensor, knn_valid: torch.Tensor):
        """
        按环切分（基于 TOP-K 顺序）
        return: lists of (idx_r, dist_r, valid_r), 每个是 (N, K_r)
        """
        rings_idx, rings_dist, rings_valid = [], [], []
        start = 0
        for size in self.ring_sizes:
            end = start + size
            rings_idx.append(knn_idx[:, start:end])
            rings_dist.append(knn_dist[:, start:end])
            rings_valid.append(knn_valid[:, start:end])
            start = end
        return rings_idx, rings_dist, rings_valid

    def _make_rel_and_pe(
        self, center_coords: torch.Tensor, neigh_coords: torch.Tensor, neigh_dist: torch.Tensor
    ):
        # rel = [dx, dy, dt, dist]
        rel_pos = neigh_coords - center_coords.unsqueeze(1)  # (N, K_r, 3)
        rel_dist = neigh_dist.unsqueeze(-1)                  # (N, K_r, 1)
        rel_encoding = torch.cat([rel_pos, rel_dist], dim=-1)  # (N, K_r, 4)
        pos_encoding = self.pos_encoder(rel_encoding)          # (N, K_r, D)
        return pos_encoding

    def _ring_aggregate(
        self, base_feat: torch.Tensor, neigh_feat: torch.Tensor, pos_enc: torch.Tensor, valid: torch.Tensor
    ):
        """
        对单个 ring 做一次 attention 聚合
        base_feat:   (N, D)          被聚合的“中心”特征
        neigh_feat:  (N, K_r, D)     邻居特征（已加 pos 编码）
        valid:       (N, K_r)        True=有效，False=padding
        return: agg: (N, D)
        """
        has_neighbors = valid.any(dim=1)  # (N,)
        agg = base_feat.new_zeros(base_feat.shape)  # (N,D)
        if has_neighbors.any():
            pts = torch.where(has_neighbors)[0]
            q = base_feat[pts].unsqueeze(1)        # (M,1,D)
            k = neigh_feat[pts]                    # (M,K_r,D)
            v = neigh_feat[pts]                    # (M,K_r,D)
            # key_padding_mask: True 表示“不要看”
            key_padding_mask = ~valid[pts]         # (M,K_r)
            out, _ = self.attention(q, k, v, key_padding_mask=key_padding_mask)
            agg[pts] = out.squeeze(1)
        return agg

    def _gate_weights(
        self,
        knn_dist: torch.Tensor,  # (N, K_total)
        knn_valid: torch.Tensor, # (N, K_total)
        center_coords: torch.Tensor,
        neigh_coords_all: torch.Tensor  # (N, K_total, 3)
    ):
        """
        生成每个点的 ring 权重（softmax），根据局部密度/时间结构
        dens_proxy: 用第 K_total 邻居距离（或均值）作为“稀疏度代理”
        time stats: |dt| 的 mean/std
        valid_ratio: 有效邻居占比
        """
        N, Kt = knn_dist.shape
        # 密度代理：用最后一个（最远）有效邻居距离（更鲁棒）
        # 若末尾无效，则用有效部分的最大值；若全无效，则置 0
        valid_counts = knn_valid.sum(dim=1).clamp(min=1)               # (N,)
        dens_proxy = torch.gather(knn_dist, 1, (valid_counts-1).unsqueeze(1)).squeeze(1)  # (N,)
        dens_proxy_norm = self._safe_norm(dens_proxy.unsqueeze(1)).squeeze(1)             # (N,)

        # 有效比率
        valid_ratio = (valid_counts.float() / float(Kt)).clamp(0, 1)   # (N,)

        gate_feats = [dens_proxy_norm, valid_ratio]

        if self.gate_use_time:
            dt = torch.abs(neigh_coords_all[..., 2] - center_coords[:, 2].unsqueeze(1))  # (N, Kt)
            # 只统计有效邻居
            dt_masked = torch.where(knn_valid, dt, torch.zeros_like(dt))
            denom = knn_valid.sum(dim=1, keepdim=True).clamp(min=1)
            dt_mean = (dt_masked.sum(dim=1, keepdim=True) / denom).squeeze(1)  # (N,)
            # std
            dt_sq = torch.where(knn_valid, (dt - dt_mean.unsqueeze(1))**2, torch.zeros_like(dt))
            dt_std = torch.sqrt((dt_sq.sum(dim=1, keepdim=True) / denom).squeeze(1) + 1e-8)  # (N,)
            # 归一化（鲁棒）
            dt_mean_n = self._safe_norm(dt_mean.unsqueeze(1)).squeeze(1)
            dt_std_n  = self._safe_norm(dt_std.unsqueeze(1)).squeeze(1)
            gate_feats += [dt_mean_n, dt_std_n]

        gate_in = torch.stack(gate_feats, dim=-1)  # (N, Fg)
        logits = self.ring_gate(gate_in)           # (N, num_rings)
        # softmax 但要 mask 掉完全无邻居的 ring（例如半径过滤导致）
        # 先统计每个 ring 的 valid 情况
        # 这里让调用方传入每个 ring 的 valid，然后做 masked softmax；为了简单，这里返回 logits，外面再 masked softmax
        return logits

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        chunk_idx: Optional[torch.Tensor] = None,
        hist_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward with ring multi-scale + soft gating (可开关)
        features: (B,N,Fin)
        coords:   (B,N,3)  -> (x,y,t)
        """
        B, N, _ = features.shape

        feat_proj = self.feat_proj(features)  # (B,N,D)
        outs = []

        for b in range(B):
            bf = feat_proj[b]    # (N,D)
            bc = coords[b]       # (N,3)

            # 是否因果
            use_causal = self.causal and (chunk_idx is None or chunk_idx[b].item() > 0)

            # KNN（一次取 K_total）
            knn_idx, knn_dist, knn_valid = self.find_knn(bc, self.k_total, causal=use_causal)  # (N,Kt)
            neigh_all = bf[knn_idx]                   # (N,Kt,D)
            neigh_coords_all = bc[knn_idx]            # (N,Kt,3)

            # 切 ring
            rings_idx, rings_dist, rings_valid = self._split_rings(knn_idx, knn_dist, knn_valid)

            # 先对所有邻居做 pos enc（每个 ring 都要用）
            pos_all = self._make_rel_and_pe(bc, neigh_coords_all, knn_dist)  # (N,Kt,D)
            neigh_all = neigh_all + pos_all

            # 每个 ring 做一次聚合
            ring_aggs: List[torch.Tensor] = []
            start = 0
            for size, valid_r in zip(self.ring_sizes, rings_valid):
                end = start + size
                if size > 0:
                    neigh_feat_r = neigh_all[:, start:end, :]   # (N,Kr,D)
                    valid_r = valid_r                           # (N,Kr)
                    agg_r = self._ring_aggregate(bf, neigh_feat_r, None, valid_r)  # (N,D)
                else:
                    agg_r = bf.new_zeros(bf.shape)
                ring_aggs.append(agg_r)
                start = end

            if self.num_rings == 1:
                enhanced = self.norm(bf + self.dropout(ring_aggs[0]))
                outs.append(enhanced)
                continue

            # 生成 ring 权重（对每个点）
            logits = self._gate_weights(knn_dist, knn_valid, bc, neigh_coords_all)  # (N, num_rings)

            # 对没有任何邻居的 ring 做 mask（若某 ring 完全无邻居，对应点上的该 ring 权重置 -inf）
            ring_valid_any = []
            for rv in rings_valid:
                ring_valid_any.append(rv.any(dim=1))   # (N,)
            ring_valid_any = torch.stack(ring_valid_any, dim=1)  # (N, num_rings) True=有邻居

            # masked softmax
            mask = ~ring_valid_any  # True 表示无效
            # 把无效位置设为极小
            logits_masked = logits.masked_fill(mask, float('-inf'))
            # 若一整行全是 -inf（极端：该点没有任何邻居），softmax 会是 nan；用 fill_value 处理
            # 这里给一个退化：如果全无邻居，则默认 near=1，其余0
            no_neighbor = mask.all(dim=1)  # (N,)
            if no_neighbor.any():
                # 把对应行的 logits 置零（softmax 后均匀），再手工设 near=1
                logits_masked = logits_masked.clone()
                logits_masked[no_neighbor] = 0.0
                # near 索引=0（按 ring_sizes 顺序）；如果不是 0，按你的 ring 定义调整
                logits_masked[no_neighbor, 0] = 10.0

            weights = torch.softmax(logits_masked, dim=1)  # (N, num_rings)

            # 加权融合
            agg_stack = torch.stack(ring_aggs, dim=1)      # (N, num_rings, D)
            fused = torch.einsum('nr,nrd->nd', weights, agg_stack)  # (N,D)

            enhanced = self.norm(bf + self.dropout(fused))
            outs.append(enhanced)

        return torch.stack(outs, dim=0)  # (B,N,D)


def create_spatial_encoder(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create spatial encoders
    """
    encoder_type = config.get('spatial_encoder_type', 'knn')

    if encoder_type == 'knn':
        knn_cfg = config.get('knn', {})
        # 新增的 ring 配置从 config 读取（可不配，保持原样）
        return KNNSpatialEncoder(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            k_neighbors=knn_cfg.get('k_neighbors', 16),
            spatial_radius=knn_cfg.get('spatial_radius', 50.0),
            aggregation='attention',
            causal=knn_cfg.get('causal', True),
            dropout=config.get('dropout', 0.1),
            use_rings=knn_cfg.get('use_rings', False),
            k_total=knn_cfg.get('k_total', None),
            ring_sizes=knn_cfg.get('ring_sizes', None),   # e.g., [16,16,16]
            gate_use_time=knn_cfg.get('gate_use_time', True),
            gate_hidden=knn_cfg.get('gate_hidden', 32)
        )
    elif encoder_type == 'knn_base':
        return BaseKNNSpatialEncoder(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            k_neighbors=config.get('knn', {}).get('k_neighbors', 16),
            spatial_radius=config.get('knn', {}).get('spatial_radius', 50.0),
            aggregation='attention',  # Fixed to attention for fair comparison
            causal=config.get('knn', {}).get('causal', True),
            dropout=config.get('dropout', 0.1)
        )
    elif encoder_type == 'grid':
        return GridSpatialEncoder(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            grid_size=(8, 8),
            neighborhood=1,
            aggregation='attention',
            dropout=config.get('dropout', 0.1)
        )
    elif encoder_type == 'voxel':
        return MicroVoxelSpatialEncoder(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            voxel_size=(2.0, 2.0, 0.005),
            neighborhood_size=1,
            aggregation='attention',
            dropout=config.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown spatial encoder type: {encoder_type}")
