"""
核心训练模块 - 完全修复版
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 导入核心模块
from core.models import build_model
from core.data.spatial_event_dataset import SpatialEventDataset, collate_fn
from core.training.losses import CombinedLoss


class Trainer:
    """统一训练器 - 支持主干和消融模型"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 首先设置实验目录
        self.exp_dir = Path(config.get('exp_dir', f'experiments/exp_{datetime.now():%Y%m%d_%H%M%S}'))
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 然后设置日志
        self.setup_logging()
        
        # 设置种子
        self.set_seed(config.get('seed', 42))
        
        # 保存配置
        self.save_config(config, self.exp_dir / 'config.json')
        
        # 初始化组件
        self.setup_data()
        self.setup_model()
        self.setup_training()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def setup_logging(self):
        """设置日志 - 确保实验目录已存在"""
        # 创建logger
        self.logger = logging.getLogger(f'trainer_{id(self)}')
        self.logger.setLevel(logging.INFO)
        
        # 清除现有handlers
        self.logger.handlers = []
        
        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件handler
        file_handler = logging.FileHandler(self.exp_dir / 'training.log')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 防止重复日志
        self.logger.propagate = False
    
    def set_seed(self, seed):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    
    def save_config(self, config, path):
        """保存配置"""
        import json
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def setup_data(self):
        """设置数据集和数据加载器"""
        self.logger.info("Loading datasets...")
        
        dataset_config = self.config['dataset']
        feature_config = self.config['features']
        
        # 归一化统计路径
        norm_stats_path = dataset_config.get('norm_stats_path', 'norm_stats.json')
        
        # 创建训练数据集
        self.train_dataset = SpatialEventDataset(
            root_dir=dataset_config['train_dir'],
            feature_config=feature_config,
            causal_history_len=dataset_config.get('causal_history_len', 0),
            norm_stats_path=norm_stats_path,
            **{k: v for k, v in dataset_config.items() 
               if k not in ['train_dir', 'val_dir', 'test_dir', 'causal_history_len', 'norm_stats_path']}
        )
        
        # 创建验证数据集
        self.val_dataset = SpatialEventDataset(
            root_dir=dataset_config['val_dir'],
            feature_config=feature_config,
            causal_history_len=dataset_config.get('causal_history_len', 0),
            norm_stats_path=norm_stats_path,
            **{k: v for k, v in dataset_config.items() 
               if k not in ['train_dir', 'val_dir', 'test_dir', 'causal_history_len', 'norm_stats_path']}
        )
        
        # 创建数据加载器
        batch_size = self.config['training']['batch_size']
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        self.logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        self.logger.info(f"Feature dimension: {self.train_dataset.get_feature_dim()}")
    
    def setup_model(self):
        """设置模型"""
        input_dim = self.train_dataset.get_feature_dim()
        model_config = self.config['model'].copy()
        model_config['input_dim'] = input_dim
        
        # 根据模型类型构建
        self.model = build_model(model_config).to(self.device)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model type: {model_config.get('type', 'Unknown')}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_training(self):
        """设置训练组件"""
        training_config = self.config['training']
        
        # 损失函数
        focal_config = {
            'alpha': training_config['focal_loss_alpha'],
            'gamma': training_config['focal_loss_gamma'],
            'ignore_index': -100
        }
        self.criterion = CombinedLoss(focal_config=focal_config)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    def calculate_accuracy(self, logits, labels):
        """计算准确率"""
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            valid_mask = (labels != -100)
            if valid_mask.any():
                correct = (preds[valid_mask] == labels[valid_mask]).sum().item()
                total = valid_mask.sum().item()
                return correct / total
            return 0.0
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        train_loss = 0
        train_acc = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch in progress_bar:
            features = batch['features'].to(self.device)
            coords = batch['coords'].to(self.device)
            labels = batch['labels'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            frame_indices = batch.get('frame_indices')
            chunk_idx = batch.get('chunk_idx')
            hist_len = batch.get('hist_len')
            
            if frame_indices is not None:
                frame_indices = frame_indices.to(self.device)
            if chunk_idx is not None:
                chunk_idx = chunk_idx.to(self.device)
            if hist_len is not None:
                hist_len = hist_len.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                # 前向传播
                output = self.model(features, coords, frame_indices, labels, valid_mask, chunk_idx, hist_len)
                logits = output['logits']
                
                # 计算损失
                loss = self.criterion(logits, labels, coords, valid_mask)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
                
                with torch.no_grad():
                    acc = self.calculate_accuracy(logits, labels)
                
                train_loss += loss.item()
                train_acc += acc
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{acc:.4f}"
                })
                
            except Exception as e:
                self.logger.error(f"Error in training batch: {e}")
                continue
        
        return train_loss / max(num_batches, 1), train_acc / max(num_batches, 1)
    
    def evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                features = batch['features'].to(self.device)
                coords = batch['coords'].to(self.device)
                labels = batch['labels'].to(self.device)
                valid_mask = batch['valid_mask'].to(self.device)
                frame_indices = batch.get('frame_indices')
                chunk_idx = batch.get('chunk_idx')
                hist_len = batch.get('hist_len')
                
                if frame_indices is not None:
                    frame_indices = frame_indices.to(self.device)
                if chunk_idx is not None:
                    chunk_idx = chunk_idx.to(self.device)
                if hist_len is not None:
                    hist_len = hist_len.to(self.device)
                
                try:
                    output = self.model(features, coords, frame_indices, labels, valid_mask, chunk_idx, hist_len)
                    logits = output['logits']
                    
                    loss = self.criterion(logits, labels, coords, valid_mask)
                    acc = self.calculate_accuracy(logits, labels)
                    
                    total_loss += loss.item()
                    total_acc += acc
                    num_batches += 1
                    
                except Exception as e:
                    self.logger.error(f"Error in evaluation batch: {e}")
                    continue
        
        return total_loss / max(num_batches, 1), total_acc / max(num_batches, 1)
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'feature_dim': self.train_dataset.get_feature_dim(),
        }
        torch.save(checkpoint, self.exp_dir / filename)
    
    def train(self):
        """主训练循环"""
        self.logger.info("Starting training...")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.evaluate(self.val_loader)
            
            # 调整学习率
            self.scheduler.step(val_loss)
            
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                self.logger.info(f"Saved best model with val_loss={val_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
        
        self.logger.info("Training completed!")