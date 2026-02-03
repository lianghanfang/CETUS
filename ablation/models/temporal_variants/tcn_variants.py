"""
TCN时序模型变体 - Temporal Convolutional Networks
用于消融实验中的时序建模对比
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TCNBlock(nn.Module):
    """TCN基础块 - 因果一维空洞卷积"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        # 因果卷积：padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """因果卷积前向传播"""
        residual = x
        
        # 第一层
        out = self.conv1(x)
        out = out[..., :x.size(-1)]  # 因果：去除未来部分
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第二层
        out = self.conv2(out)
        out = out[..., :x.size(-1)]  # 因果
        out = self.norm2(out)
        out = self.dropout(out)
        
        # 残差
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        return self.relu(out + residual)


class TCNTemporalModel(nn.Module):
    """TCN时序模型"""
    def __init__(
        self,
        d_model: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilations: Optional[List[int]] = None
    ):
        super().__init__()
        self.d_model = d_model
        
        if dilations is None:
            dilations = [2 ** i for i in range(num_layers)]
        
        layers = []
        for i, dilation in enumerate(dilations):
            layers.append(
                TCNBlock(
                    d_model, d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        self.tcn = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # (B, L, D) -> (B, D, L)
        x = x.transpose(1, 2)
        x = self.tcn(x)
        # (B, D, L) -> (B, L, D)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x