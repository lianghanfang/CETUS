## core/training/__init__.py
from .train import Trainer
from .losses import FocalLoss, CombinedLoss

__all__ = ['Trainer', 'FocalLoss', 'CombinedLoss']