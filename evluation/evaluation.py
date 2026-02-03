"""
完整的评估模块 - 包括时间窗口评估和触发延迟分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def evaluate_pixel_level(pred_path: str, threshold: Optional[float] = None) -> Dict:
    """
    像素级评估
    """
    # 读取预测文件
    df = pd.read_csv(pred_path, delim_whitespace=True)
    
    gt = df['gt'].values
    pred_prob = df['prob'].values
    
    if threshold is None:
        # 找最佳阈值
        thresholds = np.linspace(0, 1, 101)
        best_f1 = 0
        best_thr = 0.5
        
        for thr in thresholds:
            pred = (pred_prob >= thr).astype(int)
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        
        threshold = best_thr
    
    # 计算指标
    pred = (pred_prob >= threshold).astype(int)
    
    tp = np.sum((pred == 1) & (gt == 1))
    tn = np.sum((pred == 0) & (gt == 0))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    # 计算率
    pd_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate / Recall
    fa_rate = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # IoU
    intersection = tp
    union = tp + fp + fn
    iou = intersection / union if union > 0 else 0
    
    # F1 score
    f1 = 2 * precision * pd_rate / (precision + pd_rate) if (precision + pd_rate) > 0 else 0
    
    return {
        'threshold': threshold,
        'pd': pd_rate,
        'fa': fa_rate,
        'precision': precision,
        'recall': pd_rate,
        'accuracy': accuracy,
        'iou': iou,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def evaluate_temporal_windows_with_latency(
    pred_path: str,
    det_t: float = 0.02,
    threshold: float = 0.5
) -> Dict:
    """
    时间窗口评估 + 触发延迟分析
    """
    # 读取预测
    df = pd.read_csv(pred_path, delim_whitespace=True)
    
    # 按文件分组
    file_results = []
    all_latencies = []
    
    for file_idx in df['file_idx'].unique():
        file_df = df[df['file_idx'] == file_idx].sort_values('t')
        
        times = file_df['t'].values
        gt = file_df['gt'].values
        pred = (file_df['prob'].values >= threshold).astype(int)
        
        # 划分时间窗口
        t_start = times[0]
        t_end = times[-1]
        
        windows = []
        current_t = t_start
        window_idx = 0
        
        while current_t < t_end:
            window_end = current_t + det_t
            
            # 找窗口内的点
            mask = (times >= current_t) & (times < window_end)
            if mask.any():
                window_times = times[mask]
                window_gt = gt[mask]
                window_pred = pred[mask]
                
                # 窗口级别判断
                has_gt = window_gt.any()
                has_pred = window_pred.any()
                
                # 窗口中心时间
                window_center = (current_t + window_end) / 2
                
                windows.append({
                    'window_idx': window_idx,
                    't_start': current_t,
                    't_end': window_end,
                    't_center': window_center,
                    'has_gt': has_gt,
                    'has_pred': has_pred,
                    'first_gt_time': window_times[window_gt.astype(bool)][0] if has_gt else None,
                    'first_pred_time': window_times[window_pred.astype(bool)][0] if has_pred else None
                })
            
            current_t = window_end
            window_idx += 1
        
        # 计算文件级指标
        if windows:
            tp = sum(1 for w in windows if w['has_gt'] and w['has_pred'])
            fn = sum(1 for w in windows if w['has_gt'] and not w['has_pred'])
            fp = sum(1 for w in windows if not w['has_gt'] and w['has_pred'])
            tn = sum(1 for w in windows if not w['has_gt'] and not w['has_pred'])
            
            file_results.append({
                'file_idx': file_idx,
                'tp': tp,
                'fn': fn,
                'fp': fp,
                'tn': tn
            })
            
            # 计算触发延迟
            gt_objects = []
            current_obj_start = None
            
            # 识别GT对象的开始
            for i, w in enumerate(windows):
                if w['has_gt'] and current_obj_start is None:
                    # 新对象开始
                    current_obj_start = i
                elif not w['has_gt'] and current_obj_start is not None:
                    # 对象结束
                    gt_objects.append({
                        'start_window': current_obj_start,
                        'end_window': i - 1,
                        'start_time': windows[current_obj_start]['first_gt_time']
                    })
                    current_obj_start = None
            
            # 处理最后一个对象
            if current_obj_start is not None:
                gt_objects.append({
                    'start_window': current_obj_start,
                    'end_window': len(windows) - 1,
                    'start_time': windows[current_obj_start]['first_gt_time']
                })
            
            # 计算每个GT对象的检测延迟
            for obj in gt_objects:
                obj_detected = False
                detection_time = None
                
                # 在对象存在期间寻找首次检测
                for win_idx in range(obj['start_window'], obj['end_window'] + 1):
                    if windows[win_idx]['has_pred']:
                        detection_time = windows[win_idx]['first_pred_time']
                        obj_detected = True
                        break
                
                if obj_detected and detection_time is not None:
                    latency_ms = (detection_time - obj['start_time']) * 1000
                    all_latencies.append(latency_ms)
    
    # 汇总所有文件
    total_tp = sum(r['tp'] for r in file_results)
    total_fn = sum(r['fn'] for r in file_results)
    total_fp = sum(r['fp'] for r in file_results)
    total_tn = sum(r['tn'] for r in file_results)
    
    pd_rate = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    fa_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    
    # 延迟统计
    latency_stats = {}
    if all_latencies:
        all_latencies = np.array(all_latencies)
        latency_stats = {
            'mean_ms': np.mean(all_latencies),
            'median_ms': np.median(all_latencies),
            'p95_ms': np.percentile(all_latencies, 95),
            'std_ms': np.std(all_latencies),
            'min_ms': np.min(all_latencies),
            'max_ms': np.max(all_latencies)
        }
    else:
        latency_stats = {
            'mean_ms': 0,
            'median_ms': 0,
            'p95_ms': 0,
            'std_ms': 0,
            'min_ms': 0,
            'max_ms': 0
        }
    
    return {
        'det_t': det_t,
        'pd': pd_rate,
        'fa': fa_rate,
        'tp': total_tp,
        'fn': total_fn,
        'fp': total_fp,
        'tn': total_tn,
        'latency': latency_stats,
        'num_detected_objects': len(all_latencies),
        'raw_latencies': all_latencies
    }


def evaluate_temporal_windows(
    pred_path: str,
    det_t: float = 0.02,
    threshold: float = 0.5
) -> Dict:
    """
    简化的时间窗口评估（向后兼容）
    """
    result = evaluate_temporal_windows_with_latency(pred_path, det_t, threshold)
    # 移除延迟相关信息，保持接口兼容
    return {
        'det_t': result['det_t'],
        'pd': result['pd'],
        'fa': result['fa'],
        'tp': result['tp'],
        'fn': result['fn'],
        'fp': result['fp'],
        'tn': result['tn']
    }


def compute_latency_analysis(
    pred_path: str,
    det_t: float = 0.02,
    threshold: float = 0.5
) -> Dict:
    """
    专门的延迟分析函数
    """
    result = evaluate_temporal_windows_with_latency(pred_path, det_t, threshold)
    return result['latency']


def analyze_event_rate_performance(
    pred_path: str,
    rate_bins: List[float] = [0, 1000, 5000, 20000, 100000, float('inf')],
    window_size: int = 100
) -> pd.DataFrame:
    """
    分析不同事件率下的性能
    """
    df = pd.read_csv(pred_path, delim_whitespace=True)
    
    results = []
    
    for file_idx in df['file_idx'].unique():
        file_df = df[df['file_idx'] == file_idx].sort_values('t')
        
        times = file_df['t'].values
        gt = file_df['gt'].values
        pred_prob = file_df['prob'].values
        
        # 滑动窗口分析事件率
        for i in range(0, len(times), window_size):
            end_i = min(i + window_size, len(times))
            
            if end_i - i < 10:
                continue
            
            # 计算窗口事件率
            time_span = times[end_i-1] - times[i]
            if time_span > 0:
                event_rate = (end_i - i) / time_span
            else:
                continue
            
            # 确定率区间
            rate_bin = None
            for j in range(len(rate_bins) - 1):
                if rate_bins[j] <= event_rate < rate_bins[j+1]:
                    if rate_bins[j+1] == float('inf'):
                        rate_bin = f"{rate_bins[j]:.0f}+"
                    else:
                        rate_bin = f"{rate_bins[j]:.0f}-{rate_bins[j+1]:.0f}"
                    break
            
            if rate_bin is None:
                continue
            
            # 计算窗口内性能
            window_gt = gt[i:end_i]
            window_pred = (pred_prob[i:end_i] >= 0.5).astype(int)
            
            tp = np.sum((window_pred == 1) & (window_gt == 1))
            fp = np.sum((window_pred == 1) & (window_gt == 0))
            fn = np.sum((window_pred == 0) & (window_gt == 1))
            tn = np.sum((window_pred == 0) & (window_gt == 0))
            
            pd = tp / (tp + fn) if (tp + fn) > 0 else 0
            fa = fp / (fp + tn) if (fp + tn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * precision * pd / (precision + pd) if (precision + pd) > 0 else 0
            
            results.append({
                'file_idx': file_idx,
                'rate_bin': rate_bin,
                'event_rate': event_rate,
                'pd': pd,
                'fa': fa,
                'precision': precision,
                'f1': f1,
                'n_events': end_i - i,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
    
    return pd.DataFrame(results)


def compute_comprehensive_metrics(pred_path: str) -> Dict:
    """
    计算综合指标集合
    """
    results = {}
    
    # 1. 像素级评估
    results['pixel'] = evaluate_pixel_level(pred_path)
    
    # 2. 多个时间窗口评估（带延迟分析）
    temporal_windows = [0.00625, 0.0125, 0.02, 0.05]
    results['temporal'] = {}
    
    for det_t in temporal_windows:
        temp_result = evaluate_temporal_windows_with_latency(pred_path, det_t, 0.5)
        results['temporal'][f'det_{det_t}'] = temp_result
    
    # 3. 事件率性能分析
    rate_analysis = analyze_event_rate_performance(pred_path)
    if not rate_analysis.empty:
        # 按事件率区间汇总
        rate_summary = rate_analysis.groupby('rate_bin').agg({
            'pd': ['mean', 'std'],
            'fa': ['mean', 'std'],
            'f1': ['mean', 'std'],
            'event_rate': 'mean'
        }).round(4)
        
        results['event_rate_analysis'] = rate_summary.to_dict()
    
    return results


def plot_comprehensive_results(results_dict: Dict[str, Dict], output_dir: str):
    """
    绘制综合结果图表
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Pd-Fa曲线
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for (exp_name, results), color in zip(results_dict.items(), colors):
        if 'pixel' in results:
            pixel_res = results['pixel']
            plt.scatter(pixel_res['fa'], pixel_res['pd'], 
                       label=f"{exp_name} (thr={pixel_res['threshold']:.3f})",
                       color=color, s=100)
    
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Probability of Detection')
    plt.title('Pd-Fa Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(output_path / 'pd_fa_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 延迟对比
    plt.figure(figsize=(12, 6))
    exp_names = list(results_dict.keys())
    latencies_mean = []
    latencies_std = []
    
    for exp_name in exp_names:
        results = results_dict[exp_name]
        if 'temporal' in results and 'det_0.02' in results['temporal']:
            lat_stats = results['temporal']['det_0.02']['latency']
            latencies_mean.append(lat_stats['mean_ms'])
            latencies_std.append(lat_stats['std_ms'])
        else:
            latencies_mean.append(0)
            latencies_std.append(0)
    
    x_pos = np.arange(len(exp_names))
    plt.bar(x_pos, latencies_mean, yerr=latencies_std, capsize=5)
    plt.xlabel('Experiment')
    plt.ylabel('Detection Latency (ms)')
    plt.title('Detection Latency Comparison (det_t=20ms)')
    plt.xticks(x_pos, exp_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 时间分辨率曲线
    plt.figure(figsize=(12, 8))
    det_times = [0.00625, 0.0125, 0.02, 0.05]
    det_times_ms = [t * 1000 for t in det_times]
    
    for exp_name, results in results_dict.items():
        if 'temporal' in results:
            pds = []
            fas = []
            for det_t in det_times:
                key = f'det_{det_t}'
                if key in results['temporal']:
                    pds.append(results['temporal'][key]['pd'])
                    fas.append(results['temporal'][key]['fa'])
                else:
                    pds.append(0)
                    fas.append(0)
            
            plt.subplot(1, 2, 1)
            plt.semilogx(det_times_ms, pds, marker='o', label=exp_name, linewidth=2)
            plt.subplot(1, 2, 2)
            plt.semilogx(det_times_ms, fas, marker='s', label=exp_name, linewidth=2)
    
    plt.subplot(1, 2, 1)
    plt.xlabel('Detection Time Window (ms)')
    plt.ylabel('Probability of Detection')
    plt.title('Pd vs Time Window')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.xlabel('Detection Time Window (ms)')
    plt.ylabel('False Alarm Rate')
    plt.title('Fa vs Time Window')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'temporal_resolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_comprehensive_report(results_dict: Dict[str, Dict], output_path: str):
    """
    生成综合评估报告
    """
    report = []
    report.append("# Comprehensive Evaluation Report\n\n")
    
    # 1. 实验概览
    report.append("## 1. Experiment Overview\n\n")
    report.append(f"Total experiments: {len(results_dict)}\n\n")
    
    # 2. 像素级性能对比
    report.append("## 2. Pixel-level Performance\n\n")
    report.append("| Experiment | Pd | Fa | Precision | F1 | IoU | Threshold |\n")
    report.append("|------------|----|----|-----------|----|----|----------|\n")
    
    for exp_name, results in results_dict.items():
        if 'pixel' in results:
            p = results['pixel']
            report.append(f"| {exp_name} | {p['pd']:.3f} | {p['fa']:.3f} | "
                         f"{p['precision']:.3f} | {p['f1']:.3f} | {p['iou']:.3f} | "
                         f"{p['threshold']:.3f} |\n")
    
    report.append("\n")
    
    # 3. 时间窗口性能对比
    report.append("## 3. Temporal Window Performance\n\n")
    report.append("| Experiment | Pd@6.25ms | Fa@6.25ms | Pd@20ms | Fa@20ms | Latency(ms) |\n")
    report.append("|------------|-----------|-----------|---------|---------|-------------|\n")
    
    for exp_name, results in results_dict.items():
        if 'temporal' in results:
            t1 = results['temporal'].get('det_0.00625', {'pd': 0, 'fa': 0})
            t2 = results['temporal'].get('det_0.02', {'pd': 0, 'fa': 0, 'latency': {'mean_ms': 0}})
            lat = t2['latency']['mean_ms']
            
            report.append(f"| {exp_name} | {t1['pd']:.3f} | {t1['fa']:.3f} | "
                         f"{t2['pd']:.3f} | {t2['fa']:.3f} | {lat:.1f} |\n")
    
    report.append("\n")
    
    # 4. 关键发现
    report.append("## 4. Key Findings\n\n")
    
    # 找到最佳性能
    best_f1 = 0
    best_exp = ""
    best_latency = float('inf')
    best_latency_exp = ""
    
    for exp_name, results in results_dict.items():
        if 'pixel' in results:
            f1 = results['pixel']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_exp = exp_name
        
        if 'temporal' in results and 'det_0.02' in results['temporal']:
            lat = results['temporal']['det_0.02']['latency']['mean_ms']
            if lat > 0 and lat < best_latency:
                best_latency = lat
                best_latency_exp = exp_name
    
    report.append(f"- **Best F1 Score**: {best_exp} ({best_f1:.3f})\n")
    if best_latency_exp:
        report.append(f"- **Lowest Latency**: {best_latency_exp} ({best_latency:.1f} ms)\n")
    report.append("\n")
    
    # 保存报告
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    print(f"Comprehensive report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='eval_results')
    
    args = parser.parse_args()
    
    # 运行综合评估
    results = compute_comprehensive_metrics(args.pred_file)
    
    # 保存结果
    import json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'comprehensive_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Evaluation completed. Results saved to {output_dir}")
