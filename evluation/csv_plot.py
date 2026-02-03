import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_threshold_metrics(csv_files, file_names, min_threshold=0, max_threshold=1, 
                          metrics=['Pd', 'Fa', 'IoU', 'Acc'], figsize=(15, 10)):
    """
    绘制多个CSV文件中不同threshold下的各项指标曲线图
    
    Parameters:
    -----------
    csv_files : list
        CSV文件路径列表
    file_names : list  
        对应的文件名称列表（用于图例）
    min_threshold : float
        最小阈值范围
    max_threshold : float
        最大阈值范围
    metrics : list
        要绘制的指标列表
    figsize : tuple
        图像大小
    """
    
    # 定义颜色和线型
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Threshold vs Metrics Comparison', fontsize=16, fontweight='bold')
    
    # 指标对应的子图位置
    metric_positions = {
        'Pd': (0, 0),
        'Fa': (0, 1), 
        'IoU': (1, 0),
        'Acc': (1, 1)
    }
    
    # 指标的中文名称
    metric_labels = {
        'Pd': 'Detection Probability (Pd)',
        'Fa': 'False Alarm Rate (Fa)', 
        'IoU': 'Intersection over Union (IoU)',
        'Acc': 'Accuracy (Acc)'
    }
    
    # 读取并处理每个CSV文件
    for idx, (csv_file, file_name) in enumerate(zip(csv_files, file_names)):
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 确保threshold列存在
            if 'threshold' not in df.columns:
                print(f"警告: {csv_file} 中没有找到'threshold'列")
                continue
                
            # 筛选阈值范围
            df_filtered = df[(df['threshold'] >= min_threshold) & 
                           (df['threshold'] <= max_threshold)]
            
            if df_filtered.empty:
                print(f"警告: {csv_file} 在指定阈值范围内没有数据")
                continue
                
            # 按threshold排序
            df_filtered = df_filtered.sort_values('threshold')
            
            # 为每个指标绘制曲线
            for metric in metrics:
                if metric not in df_filtered.columns:
                    print(f"警告: {csv_file} 中没有找到'{metric}'列")
                    continue
                    
                if metric in metric_positions:
                    row, col = metric_positions[metric]
                    ax = axes[row, col]
                    
                    # 绘制曲线
                    line = ax.plot(df_filtered['threshold'], df_filtered[metric], 
                                 color=colors[idx % len(colors)],
                                 linestyle=line_styles[idx % len(line_styles)],
                                 linewidth=2.5,
                                 marker='o',
                                 markersize=4,
                                 label=f'{file_name}_{metric}',
                                 alpha=0.8)
                    
                    # 设置图表属性
                    ax.set_xlabel('Threshold', fontweight='bold')
                    ax.set_ylabel(metric_labels[metric], fontweight='bold')
                    ax.set_title(f'{metric_labels[metric]}', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # 设置坐标轴范围
                    ax.set_xlim(min_threshold, max_threshold)
                    
                    # 根据指标类型设置y轴范围
                    if metric in ['Pd', 'Acc', 'IoU']:
                        ax.set_ylim(0, 1.05)
                    elif metric == 'Fa':
                        ax.set_ylim(0, max(1.05, df_filtered[metric].max() * 1.1))
                        
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
    
    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def main():
    """
    主函数 - 在这里修改您的设置
    """
    
    # ================================
    # 请在这里修改您的设置
    # ================================
    
    # CSV文件路径列表（请修改为您的实际文件路径）
    csv_files = [
        r"E:\CCFA\ICASSP26\main-final\ablation_results\temporal_lstm\point_level_eval.csv",  # 请替换为您的第一个CSV文件路径
        r"E:\CCFA\ICASSP26\main-final\ablation_results\baseline\point_level_eval.csv",  # 请替换为您的第二个CSV文件路径  
        r"E:\CCFA\ICASSP26\main-final\ablation_results\temporal_gru\point_level_eval.csv"   # 请替换为您的第三个CSV文件路径
    ]
    
    # 对应的文件名称（用于图例标注）
    file_names = [
        'LSTM',      # 请替换为您想要的第一个曲线名称
        'Mamba',     # 请替换为您想要的第二个曲线名称
        'GRU' # 请替换为您想要的第三个曲线名称
    ]
    
    # 阈值范围设置
    min_threshold = 0.0   # 最小阈值
    max_threshold = 1.0   # 最大阈值
    
    # 要绘制的指标（可以选择部分指标）
    metrics_to_plot = ['Pd', 'Fa', 'IoU', 'Acc']  # 全部指标
    # metrics_to_plot = ['IoU', 'Acc']  # 或者只选择部分指标
    
    # 图像大小设置
    figure_size = (15, 10)
    
    # ================================
    # 设置结束
    # ================================
    
    # 生成图表
    try:
        fig = plot_threshold_metrics(
            csv_files=csv_files,
            file_names=file_names, 
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            metrics=metrics_to_plot,
            figsize=figure_size
        )
        
        # 显示图表
        plt.show()
        
        # 保存图表（可选）
        # fig.savefig('threshold_metrics_comparison.png', dpi=300, bbox_inches='tight')
        # print("图表已保存为 'threshold_metrics_comparison.png'")
        
    except Exception as e:
        print(f"生成图表时出错: {e}")

# 示例：如果您想要单独测试某个指标
def plot_single_metric(csv_files, file_names, metric='IoU', min_threshold=0, max_threshold=1):
    """
    绘制单个指标的对比图
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    line_styles = ['-', '--', '-.', ':', '-', '--']
    
    for idx, (csv_file, file_name) in enumerate(zip(csv_files, file_names)):
        try:
            df = pd.read_csv(csv_file)
            df_filtered = df[(df['threshold'] >= min_threshold) & 
                           (df['threshold'] <= max_threshold)]
            df_filtered = df_filtered.sort_values('threshold')
            
            plt.plot(df_filtered['threshold'], df_filtered[metric],
                    color=colors[idx % len(colors)],
                    linestyle=line_styles[idx % len(line_styles)],
                    linewidth=2.5,
                    marker='o',
                    markersize=4,
                    label=f'{file_name}_{metric}',
                    alpha=0.8)
                    
        except Exception as e:
            print(f"处理文件 {csv_file} 时出错: {e}")
    
    plt.xlabel('Threshold', fontweight='bold')
    plt.ylabel(metric, fontweight='bold') 
    plt.title(f'{metric} vs Threshold Comparison', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()