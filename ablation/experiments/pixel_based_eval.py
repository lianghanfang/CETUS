import numpy as np
import pandas as pd

def evaluate_point_level(txt_path, thresholds=np.linspace(0, 1, 101)):

    data = pd.read_csv(txt_path, sep=' ', header=0, low_memory=False)

    data['file_idx'] = data['file_idx'].astype(int)
    data['gt'] = data['gt'].astype(int)
    data['pred'] = data['pred'].astype(int)
    data['prob'] = data['prob'].astype(float)

    results = []
    for thr in thresholds:
        file_metrics = []
        for file_id, group in data.groupby('file_idx'):
            gt = group['gt'].values
            prob = group['prob'].values
            pred_thr = (prob >= thr).astype(int)

            total_bg = np.sum(gt == 0)  # 背景点数
            tp = np.sum((pred_thr == 1) & (gt == 1))
            fp = np.sum((pred_thr == 1) & (gt == 0))
            fn = np.sum((pred_thr == 0) & (gt == 1))

            Pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            Fa = fp / total_bg if total_bg > 0 else 0.0
            IoU = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            Acc = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            file_metrics.append((Pd, Fa, IoU, Acc))

        # 取所有文件的平均值
        file_metrics = np.array(file_metrics)
        Pd_avg, Fa_avg, IoU_avg, Acc_avg = file_metrics.mean(axis=0)

        results.append([thr, Pd_avg, Fa_avg, IoU_avg, Acc_avg])
        print(f"阈值={thr:.2f}  Pd={Pd_avg:.4f}  Fa={Fa_avg:.6f}  IoU={IoU_avg:.4f}  Acc={Acc_avg:.4f}")

    results_df = pd.DataFrame(results, columns=['threshold', 'Pd', 'Fa', 'IoU', 'Acc'])
    return results_df

if __name__ == '__main__':
    txt_path = 'predictions.txt'  # save.py 输出文件

    results_df = evaluate_point_level(txt_path)
    results_df.to_csv('point_level_eval.csv', index=False)