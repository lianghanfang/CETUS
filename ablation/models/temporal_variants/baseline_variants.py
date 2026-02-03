"""
基线时序模型变体 - 用于消融实验对比
"""
import torch
import torch.nn as nn


class NoTemporalModel(nn.Module):
    """无时序模型（仅空间 + 线性头）"""
    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 位置独立的线性变换"""
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x + residual
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """恒等映射"""
        return self.norm(x)


class LinearTemporalModel(nn.Module):
    """简单线性时序模型 - 每个位置独立的线性变换"""
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
        """前向传播 - 每个位置独立的线性变换"""
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x + residual