"""
RNN时序模型变体 - GRU和LSTM实现
用于消融实验中的时序建模对比
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class GRUTemporalModel(nn.Module):
    """GRU时序模型"""
    def __init__(
        self,
        d_model: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 如果是双向，需要投影回原始维度
        if bidirectional:
            self.proj = nn.Linear(2 * d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        residual = x
        x = self.norm(x)
        
        if state is not None:
            output, _ = self.gru(x, state)
        else:
            output, _ = self.gru(x)
        
        if self.bidirectional:
            output = self.proj(output)
        
        output = self.dropout(output)
        output = output + residual
        
        return output
    
    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """初始化隐藏状态"""
        num_directions = 2 if self.bidirectional else 1
        return torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.d_model,
            device=device
        )


class LSTMTemporalModel(nn.Module):
    """LSTM时序模型"""
    def __init__(
        self,
        d_model: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        if bidirectional:
            self.proj = nn.Linear(2 * d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """前向传播"""
        residual = x
        x = self.norm(x)
        
        if state is not None:
            output, _ = self.lstm(x, state)
        else:
            output, _ = self.lstm(x)
        
        if self.bidirectional:
            output = self.proj(output)
        
        output = self.dropout(output)
        output = output + residual
        
        return output
    
    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化状态"""
        num_directions = 2 if self.bidirectional else 1
        h0 = torch.zeros(
            self.num_layers * num_directions,
            batch_size,
            self.d_model,
            device=device
        )
        c0 = torch.zeros_like(h0)
        return (h0, c0)