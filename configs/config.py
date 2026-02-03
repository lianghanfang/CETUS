"""
统一配置管理模块 - 重构版
支持主干模型和消融实验模型的配置
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional


def get_main_model_config() -> Dict[str, Any]:
    """获取主干模型配置"""
    return {
        'dataset': {
            'train_dir': "./EV-UAV-dataset/EV-UAV-dataset/train",
            'val_dir': "./EV-UAV-dataset/EV-UAV-dataset/val",
            'test_dir': "./EV-UAV-dataset/EV-UAV-dataset/test",
            'chunk_size': 2000,
            'overlap': 0,
            'min_chunk_size': 2000,
            'use_chunking': True,
            'window_size': 4,
            'window_stride': 2,
            'coordinate_system': 'normalized',
            'causal_history_len': 1000,
            'norm_stats_path': 'norm_stats.json'
        },
        'features': {
            'base_features': {
                'x': True,
                'y': True,
                'timestamp': True,
                'polarity': True
            },
            'derived_features': {
                'relative_time': True,
                'polarity_encoded': True,
                'event_rate': True
            },
            'rate_window_past': 9,
            'normalization': {
                'normalize_coords': True,
                'normalize_features': True,
                'features_to_normalize': ['timestamp', 'relative_time', 'event_rate'],
                'features_no_normalize': ['polarity', 'polarity_pos', 'polarity_neg', 'x', 'y']
            }
        },
        'model': {
            'type': 'SpatialAwareEventMamba',
            'num_classes': 2,
            'hidden_dim': 128,
            'k_neighbors': 16,
            'spatial_radius': 1.0,
            'causal_knn': True,
            'mamba_blocks': 1,
            'state_dim': 32,
            'expand': 2,
            'dropout_rate': 0.1
        },
        'training': {
            'batch_size': 8,
            'num_epochs': 70,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'early_stopping_patience': 5,
            'focal_loss_alpha': 0.5,
            'focal_loss_gamma': 2.0
        },
        'inference': {
            'checkpoint_path': "best_model.pth",
            'save_txt_path': "predictions.txt",
            'threshold': 0.5,
            'chunk_size': 2000,
            'knn_history': 400,
            'use_streaming': False,
            'step_k': 1,
            'causal_knn': True,
            'device': 'cuda'
        }
    }


def get_ablation_model_config() -> Dict[str, Any]:
    """获取消融实验模型配置"""
    config = get_main_model_config()
    
    # 修改为消融模型
    config['model'] = {
        'type': 'AblationEventModel',
        'num_classes': 2,
        'hidden_dim': 64,
        'window_size': 4,
        'dropout_rate': 0.1,
        'spatial_config': {
            'spatial_encoder_type': 'knn',
            'knn': {
                'k_neighbors': 16,
                'spatial_radius': 50.0,
                'aggregation': 'attention',
                'causal': True
            }
        },
        'temporal_config': {
            'temporal_type': 'mamba',
            'num_blocks': 1,
            'mamba': {
                'd_state': 32,
                'd_conv': 4,
                'expand': 2
            }
        }
    }
    
    return config


def get_config(model_type: str = 'main') -> Dict[str, Any]:
    """
    获取配置
    
    Args:
        model_type: 模型类型，'main' 或 'ablation'
    
    Returns:
        配置字典
    """
    if model_type == 'ablation':
        return get_ablation_model_config()
    else:
        return get_main_model_config()


def save_config(config: Dict[str, Any], path: str = 'config.json'):
    """保存配置到文件"""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(path: str = 'config.json') -> Dict[str, Any]:
    """从文件加载配置"""
    with open(path, 'r') as f:
        return json.load(f)


class ConfigManager:
    """配置管理器 - 提供更高级的配置管理功能"""
    
    def __init__(self):
        self.configs = {
            'main': get_main_model_config(),
            'ablation': get_ablation_model_config()
        }
    
    def get_config(self, model_type: str = 'main') -> Dict[str, Any]:
        """获取指定类型的配置"""
        if model_type not in self.configs:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.configs[model_type].copy()
    
    def create_ablation_config(
        self, 
        spatial_type: str = 'knn',
        temporal_type: str = 'mamba',
        history_len: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """创建消融实验配置"""
        config = self.get_config('ablation')
        
        # 更新空间编码器配置
        config['model']['spatial_config']['spatial_encoder_type'] = spatial_type
        
        # 更新时序模型配置
        config['model']['temporal_config']['temporal_type'] = temporal_type
        
        # 更新历史长度
        config['dataset']['causal_history_len'] = history_len
        
        # 更新其他参数
        for key, value in kwargs.items():
            if '.' in key:
                # 支持嵌套键，如 'model.hidden_dim'
                keys = key.split('.')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[key] = value
        
        return config
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置的完整性"""
        required_sections = ['dataset', 'features', 'model', 'training', 'inference']
        
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
        
        # 验证模型类型
        model_type = config['model'].get('type')
        if model_type not in ['SpatialAwareEventMamba', 'AblationEventModel']:
            print(f"Unknown model type: {model_type}")
            return False
        
        # 验证消融模型配置
        if model_type == 'AblationEventModel':
            if 'spatial_config' not in config['model']:
                print("Missing spatial_config for AblationEventModel")
                return False
            if 'temporal_config' not in config['model']:
                print("Missing temporal_config for AblationEventModel")
                return False
        
        return True
    
    def save_config(self, config: Dict[str, Any], path: str):
        """保存配置"""
        if not self.validate_config(config):
            raise ValueError("Invalid configuration")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, path: str) -> Dict[str, Any]:
        """加载并验证配置"""
        with open(path, 'r') as f:
            config = json.load(f)
        
        if not self.validate_config(config):
            raise ValueError(f"Invalid configuration in {path}")
        
        return config


# 全局配置管理器实例
config_manager = ConfigManager()

# 便捷函数
def get_validated_config(model_type: str = 'main') -> Dict[str, Any]:
    """获取经过验证的配置"""
    return config_manager.get_config(model_type)


def create_experiment_config(
    spatial_type: str = 'knn',
    temporal_type: str = 'mamba', 
    history_len: int = 200,
    **kwargs
) -> Dict[str, Any]:
    """创建实验配置的便捷函数"""
    return config_manager.create_ablation_config(
        spatial_type=spatial_type,
        temporal_type=temporal_type,
        history_len=history_len,
        **kwargs
    )