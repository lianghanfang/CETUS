#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
per_file_eval_and_event_rate.py

读取 prediction.txt（空白分隔，列必须包含：
file_idx point_idx x y t gt pred prob）

对每个 file_idx：
1) 用原始 prob 在 0.01~1.00（100 个阈值）逐一阈值化，计算 Fa/Pd/IoU/Acc，
   保存为独立 CSV：eval/file_{fid}.csv
   - Pd = TP / (TP+FN)
   - Fa = FP / (#GT==0)
   - IoU = TP / (TP+FP+FN)
   - Acc = TP / (TP+FP)   # 与你的 eval 口径一致（相当于 precision）

2) 事件速率可视化（两联图，每文件一张）：
   - 将该文件 [t_min, t_max] 等分为 --bins 段，统计每段事件数 count；
   - 事件速率 = count / 段宽（若段宽为 0，则直接用 count）；
   - A) 时间中心 vs 事件速率；B) 段序号(1..bins) vs 事件数
   保存：plots/file_{fid}_event_rate.png

3) 事件率“云雨图”（所有文件画在一张图）：
   - 收集所有文件的“事件速率序列”（每文件长度为 --bins）
   - 在横轴上将事件速率按 --rate-bins 个区间做直方统计（全体文件采用统一的区间）
   - 行：file_idx；列：事件率区间；值：该文件在该区间中的段数
   保存：plots/event_rate_cloud_rain.png

4) 阈值热力图（只关注 Pd & Fa）：
   - 生成 Pd 值热力图：plots/threshold_pd_values.png
   - 生成 Fa 值热力图：plots/threshold_fa_values.png
   - 生成条件二值热力图（Pd≥pd_thresh）：plots/threshold_pd_mask.png
   - 生成条件二值热力图（Fa≤fa_thresh）：plots/threshold_fa_mask.png
   - 生成共同满足(Pd & Fa)热力图：plots/threshold_pd_fa_mask.png
   - 导出每文件满足阈值数量与比例：eval/threshold_condition_counts.csv

可选：
--threshold THR ：将整表按 prob>=THR 进行筛选，导出同格式 txt：
filtered/filtered_thr_{THR}.txt
其中 pred=阈值后预测，prob=阈值后概率（小于阈值的置 0）

用法：
    python per_file_eval_and_event_rate.py --input prediction.txt --outdir outputs
    python per_file_eval_and_event_rate.py --input prediction.txt --outdir outputs --threshold 0.5
    python per_file_eval_and_event_rate.py --input prediction.txt --outdir outputs --pd-thresh 0.8 --fa-thresh 0.005
"""

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能画图
import matplotlib.pyplot as plt


# ----------------------------- IO & Metrics -----------------------------
def read_predictions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件不存在：{path}")
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    needed = ["file_idx", "point_idx", "x", "y", "t", "gt", "pred", "prob"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}，需要包含列：{needed}")
    # 类型整理
    df["file_idx"] = df["file_idx"].astype(int)
    df["point_idx"] = df["point_idx"].astype(int)
    df["gt"] = df["gt"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["prob"] = df["prob"].astype(float)
    return df


def compute_metrics(gt: np.ndarray, pr: np.ndarray):
    tp = int(np.sum((pr == 1) & (gt == 1)))
    fp = int(np.sum((pr == 1) & (gt == 0)))
    fn = int(np.sum((pr == 0) & (gt == 1)))
    bg = int(np.sum(gt == 0))

    Pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0       # Recall
    Fa = fp / bg if bg > 0 else 0.0                     # False alarm / background
    IoU = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    Acc = tp / (tp + fp) if (tp + fp) > 0 else 0.0      # Precision
    return Fa, Pd, IoU, Acc


def eval_per_file(df_file: pd.DataFrame, thresholds: np.ndarray) -> pd.DataFrame:
    """对单个文件在 thresholds 上扫阈值并计算指标，返回 DataFrame。"""
    gt = df_file["gt"].values.astype(int)
    probs = df_file["prob"].values.astype(float)

    rows = []
    for thr in thresholds:
        pr = (probs >= thr).astype(int)
        Fa, Pd, IoU, Acc = compute_metrics(gt, pr)
        rows.append([thr, Fa, Pd, IoU, Acc])

    return pd.DataFrame(rows, columns=["threshold", "Fa", "Pd", "IoU", "Acc"])


# ----------------------------- Event-rate compute/plots -----------------------------
def compute_event_rates(df_file: pd.DataFrame, n_bins: int):
    """
    将 [t_min, t_max] 等分为 n_bins 段，返回：
    centers: 每段时间中心
    counts : 每段事件数量
    rates  : 每段事件速率（counts/段宽；段宽为 0 则直接 counts）
    """
    t = df_file["t"].values.astype(float)
    if len(t) == 0:
        return np.array([]), np.array([]), np.array([])
    t_min, t_max = float(np.min(t)), float(np.max(t))

    if t_max > t_min:
        edges = np.linspace(t_min, t_max, n_bins + 1)
        counts, _ = np.histogram(t, bins=edges)
        width = (t_max - t_min) / n_bins
        rates = counts / width if width > 0 else counts.astype(float)
        centers = 0.5 * (edges[:-1] + edges[1:])
    else:
        # 所有时间戳相同：退化为按索引均分
        idx = np.arange(len(t))
        bins_idx = np.linspace(0, len(t), n_bins + 1, endpoint=True)
        counts, _ = np.histogram(idx, bins=bins_idx)
        rates = counts.astype(float)
        centers = np.linspace(t_min, t_min, n_bins)

    return centers, counts, rates


def plot_event_rate_two_panel(centers, counts, rates, out_png: str, fid: int, n_bins: int):
    """两联图：A) 时间-速率；B) 段序号-事件数"""
    if len(rates) == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

    axes[0].plot(centers, rates)
    axes[0].set_title(f"A) File {fid} — Time vs Event Rate")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Events per unit time")

    axes[1].plot(np.arange(1, n_bins + 1), counts)
    axes[1].set_title(f"B) File {fid} — Bin Index vs Event Count")
    axes[1].set_xlabel(f"Bin index (1..{n_bins})")
    axes[1].set_ylabel("Event count")

    fig.suptitle(f"Event Rate Views — file_idx={fid}", y=1.02, fontsize=12)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_cloud_rain_heatmap(file_ids, per_file_rates, out_png: str,
                            rate_bins: int = 40, rate_max: float = None, pclip: float = 99.5):
    """
    云雨图：所有文件在一张热力图。
    - per_file_rates: dict[file_id] -> rates(np.ndarray, 长度=时间分段数)
    - 先将所有 rates 拼接以确定统一的事件率范围：
        * 若 rate_max 提供，则使用 [0, rate_max]
        * 否则用 [0, percentile(all_rates, pclip)] 做鲁棒上限
    - 对每个文件的 rates 在该统一范围里做直方统计（rate_bins 个柱）
    - 叠成矩阵 (n_files, rate_bins) 并 imshow
    """
    # 收集所有 rates
    all_rates = np.concatenate([r for r in per_file_rates.values() if len(r) > 0]) \
                if len(per_file_rates) else np.array([])
    if all_rates.size == 0:
        return

    # 统一的横轴范围
    if rate_max is None:
        rmax = float(np.percentile(all_rates, pclip))
        rmax = max(rmax, float(np.max(all_rates)) * 1e-6 + 1e-6)  # 防止为0
    else:
        rmax = float(rate_max)
    rmin = 0.0
    edges = np.linspace(rmin, rmax, rate_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 组装热力矩阵
    mat = []
    row_labels = []
    for fid in file_ids:
        rates = per_file_rates.get(fid, np.array([]))
        if rates.size == 0:
            hist = np.zeros(rate_bins, dtype=float)
        else:
            hist, _ = np.histogram(np.clip(rates, rmin, rmax), bins=edges)
        mat.append(hist.astype(float))
        row_labels.append(str(fid))
    mat = np.vstack(mat)  # shape: (n_files, rate_bins)

    # 画图
    fig, ax = plt.subplots(figsize=(12, max(4, len(file_ids) * 0.18)))
    im = ax.imshow(mat, aspect="auto", origin="lower",
                   extent=[centers[0], centers[-1], -0.5, len(file_ids)-0.5])

    ax.set_xlabel("Event rate (events per unit time)")
    ax.set_ylabel("file_idx")

    # y 轴刻度：防止太挤，最多显示 60 个
    if len(file_ids) <= 60:
        ax.set_yticks(range(len(file_ids)))
        ax.set_yticklabels(row_labels)
    else:
        step = int(np.ceil(len(file_ids) / 60))
        idxs = list(range(0, len(file_ids), step))
        ax.set_yticks(idxs)
        ax.set_yticklabels([row_labels[i] for i in idxs])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Time bins (count)")

    ax.set_title("Cloud-Rain Heatmap of Event Rate Distributions (all files)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)


# ----------------------------- Threshold heatmaps -----------------------------
def plot_threshold_value_heatmaps(file_ids, thresholds, pd_mat, fa_mat, out_dir):
    """
    画 Pd 值与 Fa 值的热力图（连续值），便于形状观察。
    行：file_idx；列：threshold；值：Pd 或 Fa
    """
    os.makedirs(out_dir, exist_ok=True)
    x0, x1 = float(thresholds[0]), float(thresholds[-1])

    def _plot(mat, title, fname, vmin=0.0, vmax=1.0, cmap="viridis"):
        fig, ax = plt.subplots(figsize=(12, max(4, len(file_ids) * 0.18)))
        im = ax.imshow(mat, aspect="auto", origin="lower",
                       extent=[x0, x1, -0.5, len(file_ids)-0.5],
                       vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("file_idx")
        if len(file_ids) <= 60:
            ax.set_yticks(range(len(file_ids)))
            ax.set_yticklabels([str(fid) for fid in file_ids])
        else:
            step = int(np.ceil(len(file_ids) / 60))
            idxs = list(range(0, len(file_ids), step))
            ax.set_yticks(idxs)
            ax.set_yticklabels([str(file_ids[i]) for i in idxs])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(title)
        ax.set_title(f"{title} Heatmap (files × thresholds)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=180)
        plt.close(fig)

    _plot(pd_mat, "Pd (Recall)", "threshold_pd_values.png", vmin=0.0, vmax=1.0, cmap="viridis")
    _plot(fa_mat, "Fa", "threshold_fa_values.png", vmin=0.0, vmax=float(np.nanmax(fa_mat) if fa_mat.size else 1.0), cmap="magma")


def plot_threshold_condition_heatmaps(file_ids, thresholds, pd_mat, fa_mat, pd_thresh, fa_thresh, out_dir):
    """
    画 Pd≥pd_thresh、Fa≤fa_thresh 的二值热力图，以及二者共同满足的热力图。
    """
    os.makedirs(out_dir, exist_ok=True)
    x0, x1 = float(thresholds[0]), float(thresholds[-1])

    pd_mask = (pd_mat >= pd_thresh).astype(float)
    fa_mask = (fa_mat <= fa_thresh).astype(float)
    both_mask = (pd_mask * fa_mask).astype(float)

    def _plot(mask, title, fname):
        fig, ax = plt.subplots(figsize=(12, max(4, len(file_ids) * 0.18)))
        im = ax.imshow(mask, aspect="auto", origin="lower",
                       extent=[x0, x1, -0.5, len(file_ids)-0.5],
                       vmin=0.0, vmax=1.0, cmap="Greens")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("file_idx")
        if len(file_ids) <= 60:
            ax.set_yticks(range(len(file_ids)))
            ax.set_yticklabels([str(fid) for fid in file_ids])
        else:
            step = int(np.ceil(len(file_ids) / 60))
            idxs = list(range(0, len(file_ids), step))
            ax.set_yticks(idxs)
            ax.set_yticklabels([str(file_ids[i]) for i in idxs])
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Meets condition (1=yes, 0=no)")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=180)
        plt.close(fig)

    _plot(pd_mask, f"Pd ≥ {pd_thresh} (files × thresholds)", "threshold_pd_mask.png")
    _plot(fa_mask, f"Fa ≤ {fa_thresh} (files × thresholds)", "threshold_fa_mask.png")
    _plot(both_mask, f"Pd ≥ {pd_thresh} AND Fa ≤ {fa_thresh}", "threshold_pd_fa_mask.png")

    # 统计每个文件满足阈值数量与比例
    counts = []
    for i, fid in enumerate(file_ids):
        pd_cnt = int(pd_mask[i].sum())
        fa_cnt = int(fa_mask[i].sum())
        both_cnt = int(both_mask[i].sum())
        total = pd_mask.shape[1]
        counts.append([fid, pd_cnt, pd_cnt/total, fa_cnt, fa_cnt/total, both_cnt, both_cnt/total])

    cond_df = pd.DataFrame(counts, columns=[
        "file_idx",
        f"pd_count(thr>=Pd{pd_thresh})", "pd_ratio",
        f"fa_count(thr<=Fa{fa_thresh})", "fa_ratio",
        "both_count", "both_ratio"
    ])
    return cond_df


# ----------------------------- Filtering -----------------------------
def apply_threshold_and_save(df: pd.DataFrame, thr: float, out_txt: str):
    """
    用给定阈值对 prob 做截断，导出同格式 txt：
    - pred：阈值后的二值预测 (prob>=thr -> 1 else 0)
    - prob：阈值后的概率（小于阈值的置 0，保留者保持原 prob）
    """
    out = df.copy()
    keep = (out["prob"].values >= thr)
    out.loc[~keep, "prob"] = 0.0
    out["pred"] = (keep.astype(int))

    cols = ["file_idx", "point_idx", "x", "y", "t", "gt", "pred", "prob"]
    out[cols].to_csv(out_txt, sep=" ", index=False, float_format="%.6f")


# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="按文件在 0.01~1.00 扫阈值(100份)，输出每文件 Fa/Pd/IoU/Acc、每文件事件速率图、全文件云雨图与阈值热力图。"
    )
    parser.add_argument("--input", type=str, default="predictions.txt", help="输入 txt 路径（空白分隔，含表头）")
    parser.add_argument("--outdir", type=str, default="norm_outputs", help="输出目录")
    parser.add_argument("--bins", type=int, default=100, help="时间轴分段数（用于事件速率），默认 100")
    parser.add_argument("--threshold", type=float, default=None, help="可选固定阈值，若提供则导出筛选后 txt")

    # 云雨图相关
    parser.add_argument("--rate-bins", type=int, default=40, help="云雨图横轴事件率区间个数（直方柱数），默认 40")
    parser.add_argument("--rate-max", type=float, default=None, help="云雨图横轴最大事件率；不填则用 pclip 百分位")
    parser.add_argument("--rate-pclip", type=float, default=99.5, help="云雨图横轴上限的分位裁剪（百分位），默认 99.5")

    # 阈值条件（Pd & Fa）热力图相关
    parser.add_argument("--pd-thresh", type=float, default=0.8, help="Pd 条件阈值，默认 0.8")
    parser.add_argument("--fa-thresh", type=float, default=0.005, help="Fa 条件阈值，默认 0.0050")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    eval_dir = os.path.join(args.outdir, "eval")
    plots_dir = os.path.join(args.outdir, "plots")
    filtered_dir = os.path.join(args.outdir, "filtered")
    for d in [eval_dir, plots_dir, filtered_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[1/5] 读取：{args.input}")
    df = read_predictions(args.input)

    print("[2/5] 按 file_idx 扫阈值并保存 CSV + 事件速率图")
    file_ids = sorted(df["file_idx"].unique().tolist())
    thresholds = np.linspace(0.01, 1.00, 100)

    # 收集用于热力图的 Pd / Fa 矩阵（行=文件，列=阈值）
    pd_rows = []
    fa_rows = []

    per_file_rates = {}  # 收集云雨图所需的 rates
    for fid in file_ids:
        g = df[df["file_idx"] == fid]

        # 阈值扫评估
        eval_df = eval_per_file(g, thresholds)
        eval_csv = os.path.join(eval_dir, f"file_{fid}.csv")
        eval_df.to_csv(eval_csv, index=False)
        print(f"  - 已保存评估：{eval_csv}")

        # 累积 Pd / Fa 行
        pd_rows.append(eval_df["Pd"].values.astype(float))
        fa_rows.append(eval_df["Fa"].values.astype(float))

        # 事件速率计算 & 两联图
        centers, counts, rates = compute_event_rates(g, n_bins=args.bins)
        per_file_rates[fid] = rates
        plot_path = os.path.join(plots_dir, f"file_{fid}_event_rate.png")
        plot_event_rate_two_panel(centers, counts, rates, out_png=plot_path, fid=fid, n_bins=args.bins)
        print(f"  - 已保存事件速率图：{plot_path}")

    pd_mat = np.vstack(pd_rows) if pd_rows else np.zeros((0, len(thresholds)))
    fa_mat = np.vstack(fa_rows) if fa_rows else np.zeros((0, len(thresholds)))

    print("[3/5] 生成全文件事件率云雨图（热力图）")
    cloud_path = os.path.join(plots_dir, "event_rate_cloud_rain.png")
    plot_cloud_rain_heatmap(
        file_ids=file_ids,
        per_file_rates=per_file_rates,
        out_png=cloud_path,
        rate_bins=args.rate_bins,
        rate_max=args.rate_max,
        pclip=args.rate_pclip
    )
    print(f"  - 已保存云雨图：{cloud_path}")

    print("[4/5] 生成阈值热力图（Pd/Fa 值 & 条件掩码 & 共同作用）")
    # 连续值热力图
    plot_threshold_value_heatmaps(file_ids, thresholds, pd_mat, fa_mat, plots_dir)
    # 条件热力图 + 统计
    cond_df = plot_threshold_condition_heatmaps(
        file_ids=file_ids,
        thresholds=thresholds,
        pd_mat=pd_mat,
        fa_mat=fa_mat,
        pd_thresh=args.pd_thresh,
        fa_thresh=args.fa_thresh,
        out_dir=plots_dir
    )
    cond_csv = os.path.join(eval_dir, "threshold_condition_counts.csv")
    cond_df.to_csv(cond_csv, index=False)
    print(f"  - 已保存条件统计：{cond_csv}")

    if args.threshold is not None:
        thr = float(args.threshold)
        if not (0.0 <= thr <= 1.0):
            raise ValueError("--threshold 必须在 [0,1] 范围内")
        out_txt = os.path.join(filtered_dir, f"filtered_thr_{thr:.2f}.txt")
        apply_threshold_and_save(df, thr, out_txt)
        print(f"[5/5] 已导出筛选后 txt：{out_txt}")
    else:
        print("[5/5] 未提供 --threshold，跳过导出筛选后 txt")

    print("\n完成！输出结构：")
    print(f"- {eval_dir}/file_*.csv                         # 每文件阈值-指标表（100阈值）")
    print(f"- {eval_dir}/threshold_condition_counts.csv     # 每文件满足 Pd/Fa/共同 条件的数量与比例")
    print(f"- {plots_dir}/file_*_event_rate.png             # 每文件事件速率两联图")
    print(f"- {plots_dir}/event_rate_cloud_rain.png         # 所有文件的事件率分布热力图")
    print(f"- {plots_dir}/threshold_pd_values.png           # Pd 值热力图")
    print(f"- {plots_dir}/threshold_fa_values.png           # Fa 值热力图")
    print(f"- {plots_dir}/threshold_pd_mask.png             # Pd≥pd_thresh 的二值热力图")
    print(f"- {plots_dir}/threshold_fa_mask.png             # Fa≤fa_thresh 的二值热力图")
    print(f"- {plots_dir}/threshold_pd_fa_mask.png          # 共同满足(Pd & Fa) 的二值热力图")
    if args.threshold is not None:
        print(f"- {filtered_dir}/filtered_thr_*.txt            # 筛选后同格式txt")


if __name__ == "__main__":
    main()
