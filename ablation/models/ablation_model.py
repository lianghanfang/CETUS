"""
Unified ablation experiment model - Fixed version
Supports configurable spatial encoder and temporal model combinations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

# Import factory functions with correct paths
try:
    # If running from ablation/models/
    from .spatial_variants import create_spatial_encoder
    from .temporal_variants import create_temporal_model
except ImportError:
    # If running standalone
    from ablation.models.spatial_variants import create_spatial_encoder
    from ablation.models.temporal_variants import create_temporal_model


class AblationEventModel(nn.Module):
    """
    Unified model for ablation experiments
    Flexibly combines different spatial encoders and temporal models
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        spatial_config: Optional[Dict[str, Any]] = None,
        temporal_config: Optional[Dict[str, Any]] = None,
        window_size: int = 4,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.window_size = window_size
        
        # Default configurations
        if spatial_config is None:
            spatial_config = {
                'spatial_encoder_type': 'knn',
                'knn': {
                    'k_neighbors': 16,
                    'spatial_radius': 50.0,
                    'aggregation': 'attention',
                    'causal': True
                }
            }
        
        if temporal_config is None:
            temporal_config = {
                'temporal_type': 'mamba',
                'num_blocks': 1,
                'mamba': {
                    'd_state': 32,
                    'd_conv': 4,
                    'expand': 2
                }
            }
        
        # Create spatial encoder
        spatial_config = spatial_config.copy()
        spatial_config['input_dim'] = input_dim
        spatial_config['output_dim'] = hidden_dim
        spatial_config['dropout'] = dropout_rate
        self.spatial_encoder = create_spatial_encoder(spatial_config)
        
        # Create temporal model
        temporal_config = temporal_config.copy()
        temporal_config['d_model'] = hidden_dim
        temporal_config['dropout'] = dropout_rate
        self.temporal_model = create_temporal_model(temporal_config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Store configuration info
        self.spatial_type = spatial_config.get('spatial_encoder_type', 'knn')
        self.temporal_type = temporal_config.get('temporal_type', 'mamba')
        
        # Calculate model parameters
        self.n_params = sum(p.numel() for p in self.parameters())
    
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
        """
        Forward pass
        
        Args:
            features: (B, N, input_dim) Event features
            coords: (B, N, 3) Event coordinates [x, y, t]
            frame_indices: Optional frame indices
            labels: Optional (B, N) Ground truth labels
            valid_mask: Optional (B, N) Valid point mask
            chunk_idx: Optional (B,) Chunk index for each batch
            hist_len: Optional (B,) History length for each batch
            
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        B, N, _ = features.shape
        device = features.device
        
        # Step 1: Spatial encoding (uses full sequence including history)
        enhanced_features = self.spatial_encoder(
            features, coords, valid_mask, chunk_idx, hist_len
        )
        
        # Step 2: Handle history separation for temporal modeling
        if hist_len is not None:
            # Only process current chunk through temporal model
            current_features = []
            current_lengths = []
            
            for b in range(B):
                h_len = hist_len[b].item()
                # Extract current chunk (excluding history)
                current_feat = enhanced_features[b, h_len:]
                current_features.append(current_feat)
                current_lengths.append(current_feat.size(0))
            
            # Pad to same length for batch processing
            max_len = max(current_lengths)
            padded_features = []
            
            for feat in current_features:
                if feat.size(0) < max_len:
                    padding = torch.zeros(
                        max_len - feat.size(0), feat.size(1),
                        device=device, dtype=feat.dtype
                    )
                    feat = torch.cat([feat, padding], dim=0)
                padded_features.append(feat)
            
            temporal_input = torch.stack(padded_features, dim=0)
        else:
            temporal_input = enhanced_features
            current_lengths = [N] * B
        
        # Step 3: Temporal modeling
        temporal_features = self.temporal_model(temporal_input)
        
        # Step 4: Classification
        logits = self.classifier(temporal_features)
        
        # Step 5: Reconstruct full sequence if history was present
        if hist_len is not None:
            full_logits = []
            
            for b in range(B):
                h_len = hist_len[b].item()
                curr_len = current_lengths[b]
                
                if h_len > 0:
                    # Create dummy logits for history (won't affect loss)
                    hist_logits = torch.zeros(
                        h_len, self.num_classes,
                        device=device, dtype=logits.dtype
                    )
                    curr_logits = logits[b, :curr_len]
                    full_logit = torch.cat([hist_logits, curr_logits], dim=0)
                else:
                    full_logit = logits[b, :curr_len]
                
                # Ensure correct length
                if full_logit.size(0) < N:
                    padding = torch.zeros(
                        N - full_logit.size(0), self.num_classes,
                        device=device, dtype=logits.dtype
                    )
                    full_logit = torch.cat([full_logit, padding], dim=0)
                
                full_logits.append(full_logit[:N])
            
            logits = torch.stack(full_logits, dim=0)
        
        # Step 6: Compute loss if labels provided
        output = {'logits': logits, 'features': temporal_features}
        
        if labels is not None:
            # Flatten for loss computation
            logits_flat = logits.reshape(-1, self.num_classes)
            labels_flat = labels.reshape(-1)
            
            # Only compute loss on valid labels (not -100)
            valid_labels = labels_flat != -100
            
            if valid_labels.any():
                loss = F.cross_entropy(
                    logits_flat[valid_labels],
                    labels_flat[valid_labels]
                )
                output['loss'] = loss
        
        return output
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging"""
        return {
            'spatial_type': self.spatial_type,
            'temporal_type': self.temporal_type,
            'total_params': self.n_params,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes
        }