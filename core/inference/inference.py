"""
核心推理模块 - 修复版
支持主干模型和消融实验模型的统一推理
"""
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from pathlib import Path
import gc
from typing import Optional, Tuple, List, Dict, Any

from core.models import build_model
from core.data.spatial_event_dataset import SpatialEventDataset


class InferenceEngine:
    """统一推理引擎"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        config: Optional[dict] = None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型和配置
        self.model, self.config = self.load_model(checkpoint_path)
        self.model.eval()
        
        # 如果提供了外部配置，合并推理配置
        if config:
            self.config['inference'] = {**self.config.get('inference', {}), **config.get('inference', {})}
        
        # 推理配置
        inf_cfg = self.config.get('inference', {})
        self.chunk_size = inf_cfg.get('chunk_size', 2000)
        self.knn_history = inf_cfg.get('knn_history', 400)
        self.causal_knn = inf_cfg.get('causal_knn', True)
        self.use_streaming = inf_cfg.get('use_streaming', False)
        self.step_k = inf_cfg.get('step_k', 1)
        self.threshold = inf_cfg.get('threshold', 0.5)
        
        print(f"Using device: {self.device}")
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Model type: {self.config.get('model', {}).get('type', 'Unknown')}")
        print(f"Inference mode: {'Streaming' if self.use_streaming else 'Batch'}")
    
    def load_model(self, checkpoint_path):
        """加载模型和配置"""
        print(f"Loading model from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取配置
        config = ckpt.get('config', None)
        if config is None:
            raise ValueError("No config found in checkpoint")
        
        # 构建模型
        model_config = config['model'].copy()
        model_config['input_dim'] = ckpt.get('feature_dim', 5)
        
        model = build_model(model_config)
        model = model.to(self.device)
        
        # 加载权重
        state_dict = ckpt['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        
        # 清理内存
        del ckpt, state_dict
        self.clear_memory()
        
        return model, config
    
    def clear_memory(self):
        """清理GPU和系统内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def process_sequence_chunks_with_history(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        knn_history: int,
        causal_knn: bool
    ) -> np.ndarray:
        """批量推理：跨块因果的整段前向"""
        N = len(features)
        all_probs = np.zeros(N)
        
        # 设置模型的因果模式（如果支持）
        if hasattr(self.model, 'spatial_encoder') and hasattr(self.model.spatial_encoder, 'causal'):
            self.model.spatial_encoder.causal = causal_knn
        
        # 分块处理
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            
            # 确定历史起点
            h_start = max(0, start - knn_history)
            
            # 拼接历史和当前块
            f_all = features[h_start:end]
            c_all = coords[h_start:end]
            
            # 转换为tensor
            f_tensor = torch.tensor(f_all, dtype=torch.float32).unsqueeze(0).to(self.device)
            c_tensor = torch.tensor(c_all, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 创建历史长度信息
            hist_len = torch.tensor([start - h_start], device=self.device)
            
            with torch.no_grad():
                output = self.model.forward(f_tensor, c_tensor, hist_len=hist_len)
                logits = output['logits']
                probs = F.softmax(logits, dim=-1)
                
                # 只取当前块的输出
                current_len = end - start
                hist_len_val = start - h_start
                
                if logits.dim() == 3:
                    pos_probs = probs[0, hist_len_val:hist_len_val+current_len, 1].cpu().numpy()
                else:
                    pos_probs = probs[0, 1].cpu().numpy() * np.ones(current_len)
            
            # 填充结果
            all_probs[start:end] = pos_probs
            
            self.clear_memory()
        
        return all_probs
    
    def process_sequence_streaming(
        self,
        features: np.ndarray,
        coords: np.ndarray,
        step_k: int,
        knn_history: int,
        causal_knn: bool
    ) -> np.ndarray:
        """流式推理：分块流式推进"""
        N = len(features)
        all_probs = np.zeros(N)
        
        # 检查模型是否支持流式处理
        if not hasattr(self.model, 'reset_stream') or not hasattr(self.model, 'step_block'):
            print("Warning: Model doesn't support streaming, falling back to batch mode")
            return self.process_sequence_chunks_with_history(features, coords, knn_history, causal_knn)
        
        # 初始化流式状态
        self.model.reset_stream(batch_size=1, device=self.device)
        
        # 维护历史缓存
        f_hist = []
        c_hist = []
        
        # 流式处理
        start = 0
        while start < N:
            end = min(start + step_k, N)
            actual_step = end - start
            
            # 当前块
            f_blk = features[start:end]
            c_blk = coords[start:end]
            
            # 转换为tensor
            f_blk_tensor = torch.tensor(f_blk, dtype=torch.float32).unsqueeze(0).to(self.device)
            c_blk_tensor = torch.tensor(c_blk, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 准备KNN上下文
            knn_ctx = None
            if len(f_hist) > 0:
                hist_size = min(len(f_hist), knn_history)
                f_ctx = np.array(f_hist[-hist_size:])
                c_ctx = np.array(c_hist[-hist_size:])
                
                f_ctx_tensor = torch.tensor(f_ctx, dtype=torch.float32).unsqueeze(0).to(self.device)
                c_ctx_tensor = torch.tensor(c_ctx, dtype=torch.float32).unsqueeze(0).to(self.device)
                knn_ctx = (f_ctx_tensor, c_ctx_tensor)
            
            with torch.no_grad():
                # 调用流式接口
                logits_blk = self.model.step_block(
                    f_blk_tensor, c_blk_tensor, 
                    knn_ctx=knn_ctx, block_len=actual_step
                )
                
                probs_blk = F.softmax(logits_blk, dim=-1)
                if probs_blk.dim() == 2:
                    pos_probs = probs_blk[:, 1].cpu().numpy()
                else:
                    pos_probs = np.full(actual_step, probs_blk[1].item())
            
            # 累积结果
            all_probs[start:end] = pos_probs
            
            # 更新历史缓存
            f_hist.extend(f_blk)
            c_hist.extend(c_blk)
            
            # 限制历史大小
            if len(f_hist) > knn_history * 2:
                f_hist = f_hist[-knn_history:]
                c_hist = c_hist[-knn_history:]
            
            # 定期清理内存
            if (start // step_k) % 100 == 0:
                self.clear_memory()
            
            start = end
        
        return all_probs
    
    def process_file(
        self,
        file_path: str,
        dataset_helper: SpatialEventDataset
    ) -> Optional[dict]:
        """处理单个文件"""
        try:
            # 加载数据
            data = dataset_helper._load_file(file_path)
            
            if data.shape[0] == 0 or data.shape[1] < 5:
                return None
            
            # 提取特征和坐标
            coords_original = data[:, :3].copy()
            labels = data[:, 4].astype(int) if data.shape[1] > 4 else np.zeros(len(data), dtype=int)
            features, coords_normalized = dataset_helper._extract_features(data)
            
            # 选择推理路径
            if self.use_streaming:
                pos_probs = self.process_sequence_streaming(
                    features, coords_normalized,
                    self.step_k, self.knn_history, self.causal_knn
                )
            else:
                pos_probs = self.process_sequence_chunks_with_history(
                    features, coords_normalized,
                    self.knn_history, self.causal_knn
                )
            
            # 二值化预测
            predictions = (pos_probs >= self.threshold).astype(np.uint8)
            
            return {
                'coords_original': coords_original,
                'coords_normalized': coords_normalized,
                'labels': labels,
                'probabilities': pos_probs,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    def process_batch(
        self,
        test_dir: str,
        output_path: str,
        feature_config: dict,
        dataset_config: dict
    ):
        """批量处理文件"""
        # 获取文件列表
        test_dir = Path(test_dir)
        test_files = list(test_dir.glob('*.np[yz]'))
        test_files.sort()
        print(f"Found {len(test_files)} test files")
        
        # 创建数据集辅助对象
        dataset_helper = SpatialEventDataset(
            root_dir=str(test_dir),
            feature_config=feature_config,
            chunk_size=dataset_config['chunk_size'],
            overlap=dataset_config['overlap'],
            min_chunk_size=dataset_config['min_chunk_size'],
            use_chunking=False,
            window_size=dataset_config['window_size'],
            window_stride=dataset_config['window_stride'],
            coordinate_system=dataset_config['coordinate_system'],
            norm_stats_path=dataset_config.get('norm_stats_path', None),
        )
        
        print(f"Feature dimension: {dataset_helper.get_feature_dim()}")
        print(f"Feature names: {dataset_helper.get_feature_names()}")
        
        # 处理文件
        total_points = 0
        successful_files = 0
        
        with open(output_path, 'w') as f_out:
            f_out.write("file_idx point_idx x y t gt pred prob\n")
            
            for file_idx, file_path in enumerate(tqdm(test_files, desc="Processing files")):
                result = self.process_file(str(file_path), dataset_helper)
                
                if result is None:
                    print(f"Skipping {file_path.name} due to error")
                    continue
                
                # 只保存有效标签的点
                valid_mask = result['labels'] != -100
                
                if not valid_mask.any():
                    continue
                
                valid_coords = result['coords_original'][valid_mask]
                valid_labels = result['labels'][valid_mask]
                valid_probs = result['probabilities'][valid_mask]
                valid_preds = result['predictions'][valid_mask]
                
                # 转换标签为二进制
                valid_gt_binary = (valid_labels > 0).astype(np.uint8)
                
                # 保存结果
                for point_idx, (coord, gt, pred, prob) in enumerate(zip(
                    valid_coords, valid_gt_binary, valid_preds, valid_probs
                )):
                    x, y, t = coord
                    f_out.write(f"{file_idx} {point_idx} {x:.6f} {y:.6f} {t:.6f} "
                              f"{int(gt)} {int(pred)} {prob:.6f}\n")
                
                total_points += len(valid_coords)
                successful_files += 1
                
                # 定期清理内存
                if (file_idx + 1) % 5 == 0:
                    self.clear_memory()
        
        # 打印统计
        print(f"\nProcessing completed!")
        print(f"   - Total files: {len(test_files)}")
        print(f"   - Successful files: {successful_files}")
        print(f"   - Total valid points: {total_points}")
        print(f"   - Results saved to: {output_path}")