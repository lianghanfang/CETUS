"""
空间编码器变体模块
专注于不同的空间特征聚合方法
"""

from .knn_variants import KNNAggregationVariants, create_knn_variant
from .voxel_variants import MicroVoxelSpatialEncoder
from .grid_variants import GridSpatialEncoder


def create_spatial_encoder(config):
    """
    空间编码器工厂函数
    
    Args:
        config: 包含spatial_encoder_type的配置字典
    
    Returns:
        对应的空间编码器实例
    """
    encoder_type = config.get('spatial_encoder_type', 'knn')
    
    if encoder_type == 'knn':
        from core.models.spatial_encoder import SpatialKNNEncoder
        return SpatialKNNEncoder(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            **config.get('knn', {})
        )
    elif encoder_type.startswith('knn_'):
        # KNN聚合方法变体
        variant_type = encoder_type.replace('knn_', '')
        return create_knn_variant(
            variant_type=variant_type,
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            **config.get('knn', {})
        )
    elif encoder_type == 'micro_voxel':
        return MicroVoxelSpatialEncoder(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            **config.get('micro_voxel', {})
        )
    elif encoder_type == 'grid':
        return GridSpatialEncoder(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            **config.get('grid', {})
        )
    else:
        raise ValueError(f"Unknown spatial encoder type: {encoder_type}")


__all__ = [
    'create_spatial_encoder',
    'KNNAggregationVariants',
    'create_knn_variant',
    'MicroVoxelSpatialEncoder',
    'GridSpatialEncoder'
]