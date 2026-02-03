"""
损失函数模块 - 重构版
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=-100, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            ignore_index=self.ignore_index
        )
        valid = (targets != self.ignore_index)
        if valid.any():
            ce_valid = ce_loss[valid]
            pt = torch.exp(-ce_valid)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_valid
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
        else:
            return torch.zeros((), dtype=inputs.dtype, device=inputs.device)


class CombinedLoss(nn.Module):
    """组合损失函数（简化版，仅包含Focal Loss）"""
    def __init__(self, focal_config):
        super().__init__()
        self.focal_loss = FocalLoss(**focal_config)

    def forward(self, logits, labels, coords=None, valid_mask=None):
        """
        计算损失
        
        Args:
            logits: 模型输出
            labels: 真实标签
            coords: 坐标（保留接口兼容性）
            valid_mask: 有效点掩码（保留接口兼容性）
        """
        # Focal损失
        if logits.dim() == 3:  # per_point模式
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)
            focal = self.focal_loss(logits_flat, labels_flat)
        else:  # per_window模式
            focal = self.focal_loss(logits, labels)
        
        return focal