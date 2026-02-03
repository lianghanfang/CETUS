#!/usr/bin/env python3
"""
训练入口脚本
支持主干模型和消融实验模型的统一训练
"""
import os
import sys
import argparse

# 添加项目根目录到路径
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.training.train import Trainer
from configs.config import get_config, load_config


def main():
    parser = argparse.ArgumentParser(description='Train Spatial Event Mamba Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model_type', type=str, 
                       choices=['main', 'ablation'], 
                       default='main',
                       help='Model type: main (core model) or ablation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--exp_dir', type=str, help='Experiment directory')
    
    # 消融实验相关参数
    parser.add_argument('--spatial_type', type=str, default='knn',
                       help='Spatial encoder type for ablation')
    parser.add_argument('--temporal_type', type=str, default='mamba',
                       help='Temporal model type for ablation')
    parser.add_argument('--history_len', type=int, default=200,
                       help='Causal history length for ablation')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = get_config()
    
    # 根据模型类型设置配置
    if args.model_type == 'ablation':
        # 消融实验模型配置
        config['model']['type'] = 'AblationEventModel'
        
        # 设置空间编码器配置
        config['model']['spatial_config'] = {
            'spatial_encoder_type': args.spatial_type,
            'knn': {
                'k_neighbors': 16,
                'spatial_radius': 50.0,
                'aggregation': 'attention',
                'causal': True
            }
        }
        
        # 设置时序模型配置
        config['model']['temporal_config'] = {
            'temporal_type': args.temporal_type,
            'num_blocks': 1
        }
        
        # 设置历史长度
        config['dataset']['causal_history_len'] = args.history_len
    else:
        # 主干模型配置
        config['model']['type'] = 'SpatialAwareEventMamba'
    
    # 更新命令行参数
    if args.seed:
        config['seed'] = args.seed
    if args.exp_dir:
        config['exp_dir'] = args.exp_dir
    
    print(f"Training {args.model_type} model...")
    print(f"Config: {config['model']['type']}")
    
    # 创建训练器并开始训练
    trainer = Trainer(config, device=args.device)
    trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()