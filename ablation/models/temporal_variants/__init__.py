"""
时序模型变体模块
专注于不同的时序建模方法
"""

from .rnn_variants import GRUTemporalModel, LSTMTemporalModel
from .transformer_variants import CausalTransformerModel
from .tcn_variants import TCNTemporalModel, TCNBlock
from .baseline_variants import NoTemporalModel, LinearTemporalModel


def create_temporal_model(config):
    """
    时序模型工厂函数
    
    Args:
        config: 包含temporal_type的配置字典
    
    Returns:
        对应的时序模型实例
    """
    temporal_type = config.get('temporal_type', 'mamba')
    d_model = config['d_model']
    dropout = config.get('dropout', 0.1)
    
    if temporal_type == 'mamba':
        # 使用核心Mamba模块
        from core.models.mamba_blocks import StackedStatefulMambaBlocks
        return StackedStatefulMambaBlocks(
            num_blocks=config.get('num_blocks', 1),
            d_model=d_model,
            d_state=config.get('mamba', {}).get('d_state', 16),
            d_conv=config.get('mamba', {}).get('d_conv', 4),
            expand=config.get('mamba', {}).get('expand', 2),
            dropout=dropout
        )
    elif temporal_type == 'gru':
        return GRUTemporalModel(
            d_model=d_model,
            num_layers=config.get('temporal_params', {}).get('num_layers', 1),
            dropout=dropout
        )
    elif temporal_type == 'lstm':
        return LSTMTemporalModel(
            d_model=d_model,
            num_layers=config.get('temporal_params', {}).get('num_layers', 1),
            dropout=dropout
        )
    elif temporal_type == 'tcn':
        return TCNTemporalModel(
            d_model=d_model,
            num_layers=config.get('temporal_params', {}).get('num_layers', 4),
            kernel_size=config.get('temporal_params', {}).get('kernel_size', 3),
            dilations=config.get('temporal_params', {}).get('dilations', [1, 2, 4, 8]),
            dropout=dropout
        )
    elif temporal_type == 'transformer':
        return CausalTransformerModel(
            d_model=d_model,
            num_layers=config.get('temporal_params', {}).get('num_layers', 2),
            num_heads=config.get('temporal_params', {}).get('num_heads', 4),
            dropout=dropout,
            use_rope=config.get('temporal_params', {}).get('use_rope', True)
        )
    elif temporal_type == 'none':
        return NoTemporalModel(d_model)
    elif temporal_type == 'linear':
        return LinearTemporalModel(d_model, dropout)
    else:
        raise ValueError(f"Unknown temporal model type: {temporal_type}")


__all__ = [
    'create_temporal_model',
    'GRUTemporalModel',
    'LSTMTemporalModel',
    'CausalTransformerModel',
    'TCNTemporalModel',
    'NoTemporalModel',
    'LinearTemporalModel'
]