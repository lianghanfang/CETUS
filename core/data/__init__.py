## core/data/__init__.py
from .spatial_event_dataset import SpatialEventDataset, collate_fn

__all__ = ['SpatialEventDataset', 'collate_fn']