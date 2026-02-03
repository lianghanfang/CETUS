# export_pointclouds.py
import os
import argparse
import numpy as np
import pandas as pd

PLY_HEADER_BASE = """ply
format ascii 1.0
element vertex {n}
property float x
property float y
property float z
"""

PLY_COLOR_PROPS = """property uchar red
property uchar green
property uchar blue
"""

PLY_END = "end_header\n"


def write_ply(path, pts, colors=None):
    """
    写 PLY 点云。
    pts: (N,3) numpy, columns = x,y,z
    colors: (N,3) uint8 or None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        # header
        f.write(PLY_HEADER_BASE.format(n=len(pts)))
        if colors is not None:
            f.write(PLY_COLOR_PROPS)
        f.write(PLY_END)

        # data
        if colors is None:
            for x, y, z in pts:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(pts, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def prob_to_color(prob, base=0.7):
    """
    将概率映射为颜色（计算量很小的线性映射）。
    - 先把 prob 映射到 [0,1]: v = clip((prob-base)/(1-base), 0, 1)
    - 颜色简单用红-蓝两端：low -> 蓝, high -> 红
    返回 (r,g,b) uint8
    """
    v = (prob - base) / max(1e-8, (1.0 - base))
    v = np.clip(v, 0.0, 1.0)
    r = (255.0 * v)
    g = (255.0 * (1.0 - np.abs(v - 0.5) * 2.0))  # 中间略亮，端点略暗
    b = (255.0 * (1.0 - v))
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def export_pointclouds(
    txt_path: str,
    out_dir: str = "pc_out",
    select: str = "pred",         # "pred" | "prob"
    prob_thr: float = 0.5,        # 概率阈值（select="prob"时使用）
    include_bg: bool = False,     # True: 额外导出 *_bg.ply
    scale_x: float = 1.0,         # 可选缩放
    scale_y: float = 1.0,
    scale_t: float = 1.0,         # 时间维常常需要缩放到与xy量级接近
    file_prefix: str = "file"     # 输出文件前缀
):
    # 读取
    df = pd.read_csv(txt_path, sep=r"\s+", comment="#", engine="python")
    required = {"file_idx", "point_idx", "x", "y", "t", "gt", "pred", "prob"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"缺少列: {missing}")

    # 选择依据
    select = select.lower().strip()
    if select not in {"pred", "prob"}:
        raise ValueError("--select 只能是 'pred' 或 'prob'")
    if select == "prob":
        df["pred"] = (df["prob"].values >= prob_thr).astype(np.uint8)

    # 缩放
    df["xs"] = df["x"].astype(float) * scale_x
    df["ys"] = df["y"].astype(float) * scale_y
    df["ts"] = df["t"].astype(float) * scale_t

    # 上色基准
    base = 0.5 if select == "pred" else float(prob_thr)

    # 分组导出
    for fid, g in df.groupby("file_idx"):
        # ===== GT =====
        gt_mask = (g["gt"] == 1)
        gt_pts = g.loc[gt_mask, ["xs", "ys", "ts"]].to_numpy()
        if len(gt_pts) > 0:
            gt_probs = g.loc[gt_mask, "prob"].to_numpy()
            gt_colors = prob_to_color(gt_probs, base=base)
            write_ply(os.path.join(out_dir, f"{file_prefix}_{int(fid)}_gt.ply"), gt_pts, gt_colors)

        # ===== PRED =====
        pred_mask = (g["pred"] == 1)
        pred_pts = g.loc[pred_mask, ["xs", "ys", "ts"]].to_numpy()
        if len(pred_pts) > 0:
            pred_probs = g.loc[pred_mask, "prob"].to_numpy()
            pred_colors = prob_to_color(pred_probs, base=base)
            write_ply(os.path.join(out_dir, f"{file_prefix}_{int(fid)}_pred.ply"), pred_pts, pred_colors)

        # ===== BG（可选）=====
        if include_bg:
            bg_mask = (g["gt"] == 0)
            bg_pts = g.loc[bg_mask, ["xs", "ys", "ts"]].to_numpy()
            if len(bg_pts) > 0:
                # 背景也可用其概率上色（通常较低）
                bg_probs = g.loc[bg_mask, "prob"].to_numpy()
                bg_colors = prob_to_color(bg_probs, base=base)
                write_ply(os.path.join(out_dir, f"{file_prefix}_{int(fid)}_bg.ply"), bg_pts, bg_colors)

        print(
            f"[file {fid}] gt:{gt_mask.sum()} pred:{pred_mask.sum()}"
            + (f" bg:{bg_mask.sum()}" if include_bg else "")
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", type=str, default="predictions.txt", help="保存了各点预测的txt/csv")
    ap.add_argument("--out", type=str, default="pc_out", help="输出目录")
    ap.add_argument("--select", type=str, default="prob", choices=["pred", "prob"],
                    help="按 pred 列或按概率阈值筛选")
    ap.add_argument("--prob_thr", type=float, default=0.5, help="select=prob 时使用的概率阈值")
    ap.add_argument("--include_bg", action="store_true", help="也导出背景点云")
    ap.add_argument("--scale_x", type=float, default=1.0)
    ap.add_argument("--scale_y", type=float, default=1.0)
    ap.add_argument("--scale_t", type=float, default=1.0)
    ap.add_argument("--prefix", type=str, default="file")
    args = ap.parse_args()

    export_pointclouds(
        txt_path=args.txt,
        out_dir=args.out,
        select=args.select,
        prob_thr=args.prob_thr,
        include_bg=args.include_bg,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
        scale_t=args.scale_t,
        file_prefix=args.prefix,
    )
