"""
Transformer时序模型变体 - 因果Transformer实现
用于消融实验中的时序建模对比
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RoPEPositionalEncoding(nn.Module):
    """旋转位置编码 (RoPE)"""
    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.base = base
        
        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用RoPE编码"""
        seq_len = x.size(1)
        device = x.device
        
        # 生成位置索引
        position = torch.arange(seq_len, device=device).float()
        
        # 计算角度
        freqs = torch.outer(position, self.inv_freq)  # (seq_len, d_model//2)
        freqs = torch.cat([freqs, freqs], dim=-1)     # (seq_len, d_model)
        
        # 应用旋转
        cos_freqs = freqs.cos()
        sin_freqs = freqs.sin()
        
        # RoPE变换
        x_rot = self._rotate_half(x) * sin_freqs.unsqueeze(0) + x * cos_freqs.unsqueeze(0)
        
        return x_rot
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """旋转后一半特征"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class AbsolutePositionalEncoding(nn.Module):
    """绝对位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """添加位置编码"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CausalTransformerModel(nn.Module):
    """因果Transformer模型"""
    def __init__(
        self,
        d_model: int,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = None,
        dropout: float = 0.1,
        max_len: int = 5000,
        use_rope: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_rope = use_rope
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # 位置编码
        if use_rope:
            self.pos_encoding = RoPEPositionalEncoding(d_model, max_len)
        else:
            self.pos_encoding = AbsolutePositionalEncoding(d_model, dropout, max_len)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（带因果mask）"""
        B, L, D = x.shape
        
        # 添加位置编码
        if self.use_rope:
            x = self.pos_encoding(x)
        else:
            x = self.pos_encoding(x)
        
        # 创建因果mask
        causal_mask = self._generate_causal_mask(L, x.device)
        
        # Transformer前向传播
        x = self.transformer(x, mask=causal_mask)
        x = self.norm(x)
        
        return x
    
    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果注意力mask"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask