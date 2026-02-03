
import argparse
import os
import sys
import numpy as np
import pandas as pd

try:
    import cv2
except Exception as e:
    cv2 = None

def infer_xy_mode(x, y):
    """
    Decide whether (x, y) look normalized [0,1] or already pixel coordinates.
    Returns ('norm' or 'pixel').
    """
    xmax = np.nanmax(x) if x.size else 0.0
    ymax = np.nanmax(y) if y.size else 0.0
    xmin = np.nanmin(x) if x.size else 0.0
    ymin = np.nanmin(y) if y.size else 0.0
    # Heuristic: if both axes are within [-0.01, 1.01], treat as normalized.
    if (xmin >= -0.01 and ymin >= -0.01) and (xmax <= 1.01 and ymax <= 1.01):
        return 'norm'
    return 'pixel'


def to_pixel_coords(x, y, width, height, mode='auto'):
    """
    Map (x, y) to pixel indices [col, row]. If mode='auto', try to infer.
    """
    if mode == 'auto':
        mode = infer_xy_mode(x, y)
    if mode == 'norm':
        xc = np.clip(np.round(x * (width - 1)), 0, width - 1).astype(np.int32)
        yc = np.clip(np.round(y * (height - 1)), 0, height - 1).astype(np.int32)
    else:
        xc = np.clip(np.round(x), 0, width - 1).astype(np.int32)
        yc = np.clip(np.round(y), 0, height - 1).astype(np.int32)
    return xc, yc


def connected_components_mask(points_rc, shape):
    """
    Build a binary mask of shape (H, W) from points_rc = (rows, cols) and run CC.
    Return labels image and number of components (excluding background).
    """
    H, W = shape
    if points_rc[0].size == 0:
        return None, 0
    mask = np.zeros((H, W), dtype=np.uint8)
    r, c = points_rc
    mask[r, c] = 1
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    return labels, (num_labels - 1)


def eval_paper_style(df_all,
                     width=346, height=260,
                     pd_detT=0.02, correct_thresh=0.5,
                     thresholds=np.linspace(0.01, 1.00, 100),
                     xy_mode='auto'):
    """
    Reproduce paper-style Pd / Fa sweeping over thresholds.
    Pd: fraction of GT objects detected across all frames and files.
        An object (a GT CC in a time-bin) is detected if predicted positives
        inside its region cover >= correct_thresh fraction of that object's points.
    Fa: number of false positive clusters per pixel per frame:
        total #FP components / (frames * W * H).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for connected components. Please install opencv-python.")

    # required columns
    req = {"file_idx", "x", "y", "t", "gt", "prob"}
    miss = req - set(df_all.columns)
    if miss:
        raise ValueError(f"Missing columns: {sorted(list(miss))}")

    # optional obj_id
    has_obj = "obj_id" in df_all.columns

    # precompute pixel coords
    x = df_all["x"].to_numpy(dtype=float)
    y = df_all["y"].to_numpy(dtype=float)
    cols, rows = to_pixel_coords(x, y, width, height, mode=xy_mode)
    df_all = df_all.copy()
    df_all["col"] = cols
    df_all["row"] = rows

    # groups by file
    by_file = df_all.groupby("file_idx")

    results = []
    for thr in thresholds:
        obj_num = 0         # total GT objects
        correct_num = 0     # detected GT objects
        false_num = 0       # total FP components
        frame_bins_total = 0  # total #time-bins (frames)

        for fid, g in by_file:
            # per-file time range
            ts = g["t"].to_numpy(dtype=float)
            if ts.size == 0:
                continue
            tmin, tmax = float(np.min(ts)), float(np.max(ts))
            # same open-interval binning as paper code: (t > a) & (t < b)
            n_bins = int((tmax - tmin) / pd_detT) + 1
            frame_bins_total += max(n_bins, 0)

            # pre-get arrays
            gt = g["gt"].to_numpy(dtype=int)
            prob = g["prob"].to_numpy(dtype=float)
            pred = (prob >= thr).astype(np.uint8)
            rows = g["row"].to_numpy(dtype=np.int32)
            cols = g["col"].to_numpy(dtype=np.int32)

            # If obj_id is present, map directly; else derive via CC per frame from GT points
            if has_obj:
                obj = g["obj_id"].to_numpy()
            else:
                obj = None  # will be assigned per-frame

            for i in range(n_bins):
                a = tmin + i * pd_detT
                b = tmin + (i + 1) * pd_detT
                m = (ts > a) & (ts < b)
                if not np.any(m):
                    continue

                rows_f = rows[m]
                cols_f = cols[m]
                gt_f = gt[m]
                pred_f = pred[m]

                # ----- Pd (object-level) -----
                if has_obj:
                    obj_f = g.loc[m, "obj_id"].to_numpy()
                    # iterate over object ids except background (we treat 0/negatives as bg)
                    for oid in np.unique(obj_f):
                        if oid == 0:
                            continue
                        idx_obj = (obj_f == oid)
                        if not np.any(idx_obj):
                            continue
                        obj_num += 1
                        # among GT-positive points of this object, how many are predicted positive?
                        gt_pos = (gt_f[idx_obj] == 1)
                        if not np.any(gt_pos):
                            # degenerate: object with no GT-positive points in this bin
                            continue
                        cover = float(np.sum(pred_f[idx_obj] & gt_pos)) / float(np.sum(gt_pos))
                        if cover >= correct_thresh:
                            correct_num += 1
                else:
                    # cluster GT-positive pixels into components => treat each CC as an object
                    gt_rows = rows_f[gt_f == 1]
                    gt_cols = cols_f[gt_f == 1]
                    if gt_rows.size > 0:
                        labels_img, n_obj = connected_components_mask((gt_rows, gt_cols), (height, width))
                    else:
                        labels_img, n_obj = (None, 0)

                    if n_obj > 0:
                        # map each GT-positive point to its component id
                        # (row, col) index into labels_img to read component id
                        # background=0, components=1..n_obj
                        obj_ids = labels_img[gt_rows, gt_cols] if labels_img is not None else np.zeros_like(gt_rows)
                        for oid in np.unique(obj_ids):
                            if oid == 0:
                                continue
                            obj_num += 1
                            in_obj = (obj_ids == oid)  # indices into the GT-positive subset
                            # cover = (# predicted positive among those GT-positive points) / (# GT-positive points)
                            cover = float(np.sum(pred_f[gt_f == 1][in_obj])) / float(np.sum(in_obj))
                            if cover >= correct_thresh:
                                correct_num += 1

                # ----- Fa (false alarm components outside GT) -----
                fp_rows = rows_f[(gt_f == 0) & (pred_f == 1)]
                fp_cols = cols_f[(gt_f == 0) & (pred_f == 1)]
                if fp_rows.size > 0:
                    _, n_fp_comp = connected_components_mask((fp_rows, fp_cols), (height, width))
                    false_num += n_fp_comp

        # Compute Pd and Fa
        Pd = float(correct_num) / float(obj_num) if obj_num > 0 else 0.0
        Fa = float(false_num) / float(frame_bins_total * width * height) if frame_bins_total > 0 else 0.0

        results.append([thr, Pd, Fa])

    res = pd.DataFrame(results, columns=["threshold", "Pd", "Fa"])
    return res


def main():
    ap = argparse.ArgumentParser(description="Evaluate point-wise predictions with paper-style Pd/Fa.")
    ap.add_argument("--input", type=str, default="predictions.txt", help="Path to predictions txt/csv (whitespace separated).")
    ap.add_argument("--width", type=int, default=346, help="Sensor width in pixels.")
    ap.add_argument("--height", type=int, default=260, help="Sensor height in pixels.")
    ap.add_argument("--pd-detT", type=float, default=0.00625, help="Time bin size for detection Pd.")
    ap.add_argument("--correct-thresh", type=float, default=0.9, help="Coverage threshold to count an object as detected.")
    ap.add_argument("--xy-mode", type=str, default="auto", choices=["auto", "norm", "pixel"],
                    help="How to interpret x,y (normalized [0,1] or pixel). Default auto.")
    ap.add_argument("--out", type=str, default="paper_style_eval.csv", help="Where to save the sweep results.")
    args = ap.parse_args()

    if cv2 is None:
        print("ERROR: OpenCV (cv2) not found. Please `pip install opencv-python` and retry.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Read predictions (whitespace separated, with header)
    df = pd.read_csv(args.input, sep=r"\s+", engine="python")
    res = eval_paper_style(
        df,
        width=args.width, height=args.height,
        pd_detT=args.pd_detT, correct_thresh=args.correct_thresh,
        xy_mode=args.xy_mode
    )
    res.to_csv(args.out, index=False)
    print(res.head())
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
