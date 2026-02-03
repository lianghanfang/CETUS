"""
Mamba模块 - 支持分块推进和状态传递
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from typing import Optional, List, Tuple, Union


class StatefulMambaBlock(nn.Module):
    """支持状态推进的Mamba块"""
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        整段并行前向传播（训练/离线）
        Args:
            x: 输入张量 (B, L, D)
        Returns:
            y: 输出张量 (B, L, D)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual
        return x
    
    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化状态为零
        Args:
            batch_size: 批次大小
            device: 设备
        Returns:
            (hidden_state, conv_state): 初始状态元组
        """
        d_inner = self.d_model * self.expand
        hidden_state = torch.zeros(
            batch_size, d_inner, self.d_state, 
            device=device, dtype=torch.float32
        )
        conv_state = torch.zeros(
            batch_size, d_inner, self.d_conv - 1,
            device=device, dtype=torch.float32
        )
        return (hidden_state, conv_state)
    
    def forward_chunk(
        self, 
        x_chunk: torch.Tensor, 
        state: Tuple[torch.Tensor, torch.Tensor],
        block_len: int = 1
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        分块前向传播（流式推理）
        Args:
            x_chunk: 输入块 (B, Lc, D)
            state: 状态元组 (hidden_state, conv_state)
            block_len: 每步并行点数（用于延迟/吞吐折中）
        Returns:
            (y_chunk, new_state): 输出块和新状态
        """
        B, Lc, D = x_chunk.shape
        
        # 保存残差
        residual = x_chunk
        x_chunk = self.norm(x_chunk)
        
        # TODO: 这里需要mamba_ssm支持状态传递
        # 暂时使用标准forward，实际应该使用带状态的推进
        # 这需要mamba_ssm库的支持或自定义实现
        y_chunk = self.mamba(x_chunk)
        
        y_chunk = self.dropout(y_chunk)
        y_chunk = y_chunk + residual
        
        # 返回更新后的状态（这里简化处理，实际需要从mamba获取）
        new_state = state  # 简化：保持状态不变
        
        return y_chunk, new_state


class StackedStatefulMambaBlocks(nn.Module):
    """可堆叠的Mamba块序列"""
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.num_blocks = num_blocks
        
        for _ in range(num_blocks):
            block = StatefulMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            self.blocks.append(block)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        整段并行前向传播（训练/离线）
        Args:
            x: 输入 (B, L, D)
        Returns:
            y: 输出 (B, L, D)
        """
        for block in self.blocks:
            x = block.forward(x)
        return x
    
    def init_states(self, batch_size: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        初始化所有块的状态
        Args:
            batch_size: 批次大小
            device: 设备
        Returns:
            states: 状态列表
        """
        states = []
        for block in self.blocks:
            state = block.init_state(batch_size, device)
            states.append(state)
        return states
    
    def forward_chunk(
        self,
        x_chunk: torch.Tensor,
        states: List[Tuple[torch.Tensor, torch.Tensor]],
        block_len: int = 1
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        分块前向传播，块内并行、块间传状态
        Args:
            x_chunk: 输入块 (B, Lc, D)
            states: 各层状态列表
            block_len: 每步并行点数
        Returns:
            (y_chunk, new_states): 输出块和新状态列表
        """
        new_states = []
        y = x_chunk
        
        for i, block in enumerate(self.blocks):
            state = states[i] if states and i < len(states) else None
            if state is None:
                # 如果没有提供状态，初始化
                state = block.init_state(y.shape[0], y.device)
            
            y, new_state = block.forward_chunk(y, state, block_len)
            new_states.append(new_state)
        
        return y, new_states