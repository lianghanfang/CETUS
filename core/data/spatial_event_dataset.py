"""
空间事件数据集 - 支持因果特征提取和历史前缀
"""
import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path


class SpatialEventDataset(Dataset):
    """
    空间感知事件数据集
    支持因果特征提取、历史前缀、时间排序
    """

    def __init__(
            self,
            root_dir: str,
            feature_config: Optional[Dict] = None,
            chunk_size: int = 800,
            overlap: int = 100,
            min_chunk_size: int = 200,
            window_size: int = 4,
            window_stride: int = 2,
            coordinate_system: str = 'normalized',
            use_chunking: bool = True,
            causal_history_len: int = 0,  # 新增：历史前缀长度
            norm_stats_path: Optional[str] = None,  # 新增：归一化统计路径
    ):
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy') or f.endswith('.npz')]
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.window_size = window_size
        self.window_stride = window_stride
        self.coordinate_system = coordinate_system
        self.use_chunking = use_chunking
        self.causal_history_len = causal_history_len  # 历史前缀长度
        self.norm_stats_path = norm_stats_path

        # 默认特征配置
        self.default_feature_config = {
            'base_features': {
                'x': False,
                'y': False,
                'timestamp': True,
                'polarity': True
            },
            'derived_features': {
                'relative_time': True,
                'polarity_encoded': True,
                'event_rate': False,
            },
            'rate_window_past': 5,  # 因果：仅使用过去窗口
            'normalization': {
                'normalize_coords': True,
                'normalize_features': True,
                'features_to_normalize': ['timestamp', 'relative_time', 'event_rate'],
                'features_no_normalize': ['polarity', 'polarity_pos', 'polarity_neg', 'x', 'y'],
            }
        }

        # 合并用户配置
        self.feature_config = self._merge_config(feature_config)
        self.feature_names = self._build_feature_names()

        # 如果启用分块，预先计算所有chunk的索引（包含历史前缀）
        if self.use_chunking:
            print("Preparing chunk indices with causal history...")
            self.chunk_indices = self._prepare_chunk_indices_with_history()
            print(f"Generated {len(self.chunk_indices)} chunks from {len(self.file_list)} files")
        else:
            self.chunk_indices = None

        # 归一化统计：从文件加载或计算
        self.normalization_stats = {}
        if self.feature_config['normalization']['normalize_features']:
            if norm_stats_path and os.path.exists(norm_stats_path):
                print(f"Loading normalization stats from {norm_stats_path}")
                self.normalization_stats = self._load_norm_stats(norm_stats_path)
            else:
                self._compute_normalization_stats()
                if norm_stats_path:
                    self._save_norm_stats(norm_stats_path)

    def _merge_config(self, user_config: Optional[Dict]) -> Dict:
        """深度合并用户配置和默认配置"""
        if user_config is None:
            return self.default_feature_config.copy()

        def deep_merge(default, user):
            merged = default.copy()
            for key, value in user.items():
                if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = deep_merge(merged[key], value)
                else:
                    merged[key] = value
            return merged

        return deep_merge(self.default_feature_config, user_config)

    def _build_feature_names(self) -> List[str]:
        """构建特征名称列表"""
        names = []
        
        # 基础特征
        if self.feature_config['base_features'].get('x', False):
            names.append('x')
        if self.feature_config['base_features'].get('y', False):
            names.append('y')
        if self.feature_config['base_features'].get('timestamp', True):
            names.append('timestamp')
        if self.feature_config['base_features'].get('polarity', True):
            names.append('polarity')
            
        # 衍生特征
        if self.feature_config.get('derived_features', {}).get('relative_time', False):
            names.append('relative_time')
        if self.feature_config.get('derived_features', {}).get('polarity_encoded', False):
            names.extend(['polarity_pos', 'polarity_neg'])
        if self.feature_config.get('derived_features', {}).get('event_rate', False):
            names.append('event_rate')
                    
        return names

    def _load_file(self, file_path: str) -> np.ndarray:
        """加载单个文件并按时间排序"""
        try:
            data = np.load(file_path)
            if isinstance(data, np.lib.npyio.NpzFile):
                possible_keys = ['evs_norm', 'events', 'ev', 'data']
                key = None
                for k in possible_keys:
                    if k in data:
                        key = k
                        break
                if key is None:
                    key = list(data.keys())[0]
                data = data[key]
            
            # 确保按时间排序（coords[:, 2]是时间戳）
            if data.shape[0] > 0 and data.shape[1] >= 3:
                time_order = np.argsort(data[:, 2])
                data = data[time_order]
            
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.zeros((1, 6))

    def _prepare_chunk_indices_with_history(self) -> List[Tuple[int, int, int, int, int]]:
        """
        预先计算所有文件的分块索引，包含历史前缀和块序号
        返回: [(file_idx, hist_start, chunk_start, chunk_end, chunk_idx), ...]
        """
        chunk_indices = []
        
        for file_idx, filename in enumerate(self.file_list):
            try:
                file_path = os.path.join(self.root_dir, filename)
                data = self._load_file(file_path)
                
                if data.shape[0] == 0:
                    continue
                
                sequence_length = len(data)
                chunk_idx = 0  # 块序号计数器
                
                if sequence_length <= self.min_chunk_size:
                    # 短序列：整体作为一个块，无历史，块序号为0
                    chunk_indices.append((file_idx, 0, 0, sequence_length, 0))
                else:
                    step = self.chunk_size - self.overlap
                    for start_idx in range(0, sequence_length, step):
                        end_idx = min(start_idx + self.chunk_size, sequence_length)
                        
                        if end_idx - start_idx >= self.min_chunk_size:
                            # 确定历史起点
                            hist_start = max(0, start_idx - self.causal_history_len)
                            chunk_indices.append((file_idx, hist_start, start_idx, end_idx, chunk_idx))
                            chunk_idx += 1
                        
                        if end_idx >= sequence_length:
                            break
                    
                    # 确保最后的数据被包含
                    if chunk_indices and chunk_indices[-1][3] < sequence_length:
                        last_start = max(0, sequence_length - self.chunk_size)
                        if sequence_length - last_start >= self.min_chunk_size:
                            hist_start = max(0, last_start - self.causal_history_len)
                            # 注意这里也需要添加chunk_idx
                            chunk_indices.append((file_idx, hist_start, last_start, sequence_length, chunk_idx))
                
            except Exception as e:
                print(f"Warning: Failed to process {filename} for chunking: {e}")
                continue
        
        return chunk_indices

    def _compute_event_rate_causal(self, timestamps: np.ndarray) -> np.ndarray:
        """计算因果事件率特征（仅使用过去窗口）"""
        N = len(timestamps)
        window_size = self.feature_config.get('rate_window_past', 5)
        event_rates = np.zeros(N)
        
        for i in range(N):
            # 纯过去窗口 [i-w+1, i]
            start = max(0, i - window_size + 1)
            end = i + 1  # 包含当前点
            
            if end - start < 2:
                event_rates[i] = 0
                continue
            
            time_span = timestamps[end - 1] - timestamps[start]
            if time_span > 0:
                event_rates[i] = (end - start) / time_span
            else:
                event_rates[i] = 0
        
        return event_rates

    def _extract_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """提取特征和坐标（因果版本）"""
        N = len(data)
        
        # 提取坐标
        coords = data[:, :3]
        
        features_dict = {}
        
        # 基础特征
        if self.feature_config['base_features'].get('x', False):
            features_dict['x'] = coords[:, 0]
        if self.feature_config['base_features'].get('y', False):
            features_dict['y'] = coords[:, 1]
        if self.feature_config['base_features'].get('timestamp', True):
            features_dict['timestamp'] = data[:, 2]
        if self.feature_config['base_features'].get('polarity', True):
            features_dict['polarity'] = data[:, 3]
        
        # 衍生特征（因果版本）
        if self.feature_config.get('derived_features', {}).get('relative_time', False):
            # 因果：使用 t_i - t_{i-1}
            rel_time = np.zeros(N)
            if N > 1:
                rel_time[1:] = data[1:, 2] - data[:-1, 2]
            features_dict['relative_time'] = rel_time
            
        if self.feature_config.get('derived_features', {}).get('polarity_encoded', False):
            polarity = data[:, 3]
            features_dict['polarity_pos'] = (polarity > 0).astype(float)
            features_dict['polarity_neg'] = (polarity <= 0).astype(float)
        
        if self.feature_config.get('derived_features', {}).get('event_rate', False):
            # 使用因果版本的事件率
            event_rates = self._compute_event_rate_causal(data[:, 2])
            features_dict['event_rate'] = event_rates
        
        # 组装特征矩阵
        feature_list = []
        for name in self.feature_names:
            if name in features_dict:
                feature_list.append(features_dict[name])
                
        if len(feature_list) > 0:
            feature_matrix = np.stack(feature_list, axis=1)
        else:
            feature_matrix = data[:, 2:4]  # 默认使用timestamp和polarity
            
        # 特征归一化
        if self.feature_config['normalization']['normalize_features']:
            feature_matrix = self._normalize_features(feature_matrix)
            
        return feature_matrix, coords

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """选择性归一化特征"""
        normalized = features.copy()
        features_to_normalize = self.feature_config['normalization'].get('features_to_normalize', [])
        features_no_normalize = self.feature_config['normalization'].get('features_no_normalize', [])
        
        for i, name in enumerate(self.feature_names):
            # 跳过不需要归一化的特征
            if name in features_no_normalize:
                continue
            
            # 使用预计算的统计量归一化
            if name in features_to_normalize and name in self.normalization_stats:
                stats = self.normalization_stats[name]
                normalized[:, i] = (normalized[:, i] - stats['mean']) / stats['std']
            # 对于其他特征，如果不在no_normalize列表中，使用局部统计
            elif name not in features_no_normalize:
                col = normalized[:, i]
                if np.std(col) > 1e-8:
                    normalized[:, i] = (col - np.mean(col)) / (np.std(col) + 1e-8)
                    
        return normalized

    def _compute_normalization_stats(self):
        """计算归一化统计信息（在训练集上）"""
        print("Computing normalization statistics...")
        
        # 需要统计的特征
        features_to_stat = self.feature_config['normalization'].get('features_to_normalize', [])
        stats_data = {feat: [] for feat in features_to_stat}
        
        # 采样文件
        sample_size = min(50, len(self.file_list))
        for file in self.file_list[:sample_size]:
            try:
                file_path = os.path.join(self.root_dir, file)
                data = self._load_file(file_path)
                
                if data.shape[0] == 0:
                    continue
                
                # 提取需要统计的原始数据
                if 'timestamp' in stats_data:
                    stats_data['timestamp'].extend(data[:, 2])
                
                # 计算衍生特征用于统计（因果版本）
                if 'relative_time' in stats_data:
                    rel_time = np.zeros(len(data))
                    if len(data) > 1:
                        rel_time[1:] = data[1:, 2] - data[:-1, 2]
                    stats_data['relative_time'].extend(rel_time)
                
                if 'event_rate' in stats_data:
                    event_rates = self._compute_event_rate_causal(data[:, 2])
                    stats_data['event_rate'].extend(event_rates)
                    
            except Exception as e:
                print(f"Warning: Failed to load {file} for statistics: {e}")
                continue
        
        # 计算统计量
        for key, values in stats_data.items():
            if len(values) > 0:
                values = np.array(values)
                self.normalization_stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values) + 1e-8),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

    def _save_norm_stats(self, path: str):
        """保存归一化统计到文件"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.normalization_stats, f, indent=2)
        print(f"Saved normalization stats to {path}")

    def _load_norm_stats(self, path: str) -> dict:
        """从文件加载归一化统计"""
        with open(path, 'r') as f:
            return json.load(f)

    def _create_sliding_windows(self, data: np.ndarray) -> List[Tuple[int, int]]:
        """创建滑动窗口索引"""
        N = len(data)
        windows = []
        
        for start_idx in range(0, N - self.window_size + 1, self.window_stride):
            end_idx = min(start_idx + self.window_size, N)
            windows.append((start_idx, end_idx))
            
        if len(windows) == 0 or windows[-1][1] < N:
            start_idx = max(0, N - self.window_size)
            windows.append((start_idx, N))
            
        return windows

    def __len__(self):
        if self.use_chunking and self.chunk_indices is not None:
            return len(self.chunk_indices)
        else:
            return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回数据项，历史部分只用于KNN"""
        if self.use_chunking and self.chunk_indices is not None:
            # 分块模式
            file_idx, hist_start, chunk_start, chunk_end, chunk_idx = self.chunk_indices[idx]
            file_path = os.path.join(self.root_dir, self.file_list[file_idx])
            full_data = self._load_file(file_path)
            
            # 提取数据：历史前缀 + 当前块
            data = full_data[hist_start:chunk_end]
            hist_len = chunk_start - hist_start  # 历史前缀长度
        else:
            # 原始模式
            file_path = os.path.join(self.root_dir, self.file_list[idx])
            data = self._load_file(file_path)
            hist_len = 0
            chunk_idx = 0
            file_idx = idx
        
        # 确保数据有效
        if data.shape[0] == 0 or data.shape[1] < 5:
            data = np.array([[128, 128, 0, 1, 0, 0]])
            hist_len = 0
        
        # 提取特征和坐标
        features, coords = self._extract_features(data)
        
        # 标签：历史部分标记为-100（不参与损失计算）
        if data.shape[1] > 4:
            labels = data[:, 4].astype(int)
        else:
            labels = np.zeros(len(data), dtype=int)
        
        # 只有当前块的标签有效，历史部分设为-100
        if hist_len > 0:
            labels[:hist_len] = -100
        
        # 创建帧索引
        frame_indices = np.zeros(len(data), dtype=int)
        windows = self._create_sliding_windows(data)
        for frame_idx, (start, end) in enumerate(windows):
            frame_indices[start:end] = frame_idx % self.window_size
        
        result = {
            'features': torch.tensor(features, dtype=torch.float32),
            'coords': torch.tensor(coords, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'frame_indices': torch.tensor(frame_indices, dtype=torch.long),
            'chunk_idx': chunk_idx,
            'file_idx': file_idx,
            'hist_len': hist_len,  # 添加历史长度信息
        }
        
        return result

    def get_feature_dim(self) -> int:
        """返回特征维度"""
        return len(self.feature_names) if len(self.feature_names) > 0 else 2

    def get_feature_names(self) -> List[str]:
        """返回特征名称列表"""
        return self.feature_names.copy()


def collate_fn(batch):
    """批次整理函数，传递历史长度信息"""
    from torch.nn.utils.rnn import pad_sequence
    
    features_list = [item['features'] for item in batch]
    coords_list = [item['coords'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    frame_indices_list = [item.get('frame_indices', torch.zeros(len(item['features']), dtype=torch.long)) 
                         for item in batch]
    
    # 提取元信息
    chunk_idx_list = [item.get('chunk_idx', 0) for item in batch]
    file_idx_list = [item.get('file_idx', 0) for item in batch]
    hist_len_list = [item.get('hist_len', 0) for item in batch]
    
    # 填充到相同长度
    features_padded = pad_sequence(features_list, batch_first=True, padding_value=0.0)
    coords_padded = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    frame_indices_padded = pad_sequence(frame_indices_list, batch_first=True, padding_value=0)
    
    # 创建valid_mask（标识非padding位置）
    lengths = [len(f) for f in features_list]
    max_len = features_padded.size(1)
    valid_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        valid_mask[i, :length] = True
    
    return {
        'features': features_padded,
        'coords': coords_padded,
        'labels': labels_padded,
        'frame_indices': frame_indices_padded,
        'valid_mask': valid_mask,
        'chunk_idx': torch.tensor(chunk_idx_list, dtype=torch.long),
        'file_idx': torch.tensor(file_idx_list, dtype=torch.long),
        'hist_len': torch.tensor(hist_len_list, dtype=torch.long),  # 新增
    }