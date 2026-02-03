"""
Temporal model factory and variants for ablation experiments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple


def create_temporal_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create temporal models
    """
    temporal_type = config.get('temporal_type', 'mamba')
    d_model = config.get('d_model', 128)
    dropout = config.get('dropout', 0.1)
    
    if temporal_type == 'mamba':
        # Import from core or define simplified version
        try:
            from mamba_ssm import Mamba
            return SimplifiedMambaBlock(
                d_model=d_model,
                d_state=config.get('mamba', {}).get('d_state', 32),
                d_conv=config.get('mamba', {}).get('d_conv', 4),
                expand=config.get('mamba', {}).get('expand', 2),
                dropout=dropout
            )
        except ImportError:
            # Fallback to linear if mamba not available
            return LinearTemporalModel(d_model, dropout)
            
    elif temporal_type == 'gru':
        return GRUTemporalModel(
            d_model=d_model,
            num_layers=2,
            dropout=dropout
        )
    elif temporal_type == 'lstm':
        return LSTMTemporalModel(
            d_model=d_model,
            num_layers=2,
            dropout=dropout
        )
    elif temporal_type == 'transformer':
        return CausalTransformerModel(
            d_model=d_model,
            num_layers=2,
            num_heads=4,
            dropout=dropout
        )
    elif temporal_type == 'none':
        return LinearTemporalModel(d_model, dropout)
    else:
        raise ValueError(f"Unknown temporal model type: {temporal_type}")


class SimplifiedMambaBlock(nn.Module):
    """Simplified Mamba block for when mamba_ssm is available"""
    def __init__(self, d_model: int, d_state: int = 32, d_conv: int = 4, 
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        except ImportError:
            # Fallback to linear layer
            self.mamba = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual


class LinearTemporalModel(nn.Module):
    """Simple linear temporal model (baseline)"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x + residual


class GRUTemporalModel(nn.Module):
    """GRU temporal model"""
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Causal
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        if state is not None:
            output, _ = self.gru(x, state)
        else:
            output, _ = self.gru(x)
        
        output = self.dropout(output)
        return output + residual


class LSTMTemporalModel(nn.Module):
    """LSTM temporal model"""
    def __init__(self, d_model: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Causal
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, 
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        if state is not None:
            output, _ = self.lstm(x, state)
        else:
            output, _ = self.lstm(x)
        
        output = self.dropout(output)
        return output + residual


class CausalTransformerModel(nn.Module):
    """Causal Transformer model with absolute position encoding"""
    def __init__(self, d_model: int, num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Absolute position encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal masking"""
        B, L, D = x.shape
        
        # Add position encoding
        x = x + self.pe[:, :L]
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self._generate_causal_mask(L, x.device)
        
        # Transformer forward
        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)
        
        return x
    
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate upper triangular causal mask"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask