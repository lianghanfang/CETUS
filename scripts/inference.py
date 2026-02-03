#!/usr/bin/env python3
"""
推理入口脚本
"""
import os
import sys
import argparse

# 添加项目根目录到路径
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from core.inference.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description='Inference for Spatial Event Mamba')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Test dataset directory')
    parser.add_argument('--output', type=str, default='predictions.txt',
                       help='Output predictions file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    # 推理配置
    parser.add_argument('--use_streaming', action='store_true',
                       help='Use streaming inference mode')
    parser.add_argument('--chunk_size', type=int, default=2000,
                       help='Chunk size for batch inference')
    parser.add_argument('--knn_history', type=int, default=400,
                       help='KNN history context size')
    parser.add_argument('--step_k', type=int, default=1,
                       help='Step size for streaming inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    args = parser.parse_args()
    
    # 构建推理配置
    inference_config = {
        'use_streaming': args.use_streaming,
        'chunk_size': args.chunk_size,
        'knn_history': args.knn_history,
        'step_k': args.step_k,
        'threshold': args.threshold
    }
    
    print(f"Starting inference...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test dir: {args.test_dir}")
    print(f"Output: {args.output}")
    print(f"Mode: {'Streaming' if args.use_streaming else 'Batch'}")
    
    # 创建推理引擎
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        config={'inference': inference_config}
    )
    
    # 执行推理
    engine.process_batch(
        test_dir=args.test_dir,
        output_path=args.output,
        feature_config=engine.config['features'],
        dataset_config=engine.config['dataset']
    )
    
    print("Inference completed!")


if __name__ == "__main__":
    main()