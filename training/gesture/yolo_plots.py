"""YOLO-style training dashboard plots (no step_recognition imports)."""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d

SMOOTH_SIGMA = 3


def smooth(y, sigma=SMOOTH_SIGMA):
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return y
    return gaussian_filter1d(y, sigma=sigma)


def load_epoch_metrics(metrics_path):
    rows = []
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty metrics: {metrics_path}")

    ep = np.array([int(r["epoch"]) for r in rows])
    data = {"epoch": ep}
    for key in ("train_loss", "train_acc", "train_lr", "val_loss", "val_acc"):
        eps, vals = [], []
        for r in rows:
            if r.get(key, "").strip() != "":
                eps.append(int(r["epoch"]))
                vals.append(float(r[key]))
        if vals:
            data[key] = (np.array(eps), np.array(vals))
    return data


def build_metrics_from_checkpoints(ckpt_dir, out_csv):
    """從訓練過程存下的多個 .pth 還原真實 epoch 曲線。"""
    import torch

    ckpt_dir = Path(ckpt_dir)
    rows = []
    for p in sorted(ckpt_dir.glob("hand_gesture*.pth")):
        s = torch.load(p, map_location="cpu", weights_only=False)
        ep = s.get("epoch")
        if ep is None:
            continue
        rows.append({
            "epoch": int(ep) + 1,
            "train_loss": s.get("train_loss"),
            "train_acc": s.get("val_acc"),  # train.py 未單獨存 train_acc；用 val 作參考時跳過
            "val_loss": s.get("val_loss"),
            "val_acc": s.get("val_acc"),
        })

    if not rows:
        return None

    # 去重 epoch，保留該 epoch 最佳 val_acc
    by_ep = {}
    for r in rows:
        e = r["epoch"]
        if e not in by_ep or (r["val_acc"] or 0) >= (by_ep[e]["val_acc"] or 0):
            by_ep[e] = r

    fields = ["epoch", "train_loss", "train_acc", "train_lr", "val_loss", "val_acc"]
    out_csv = Path(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in sorted(by_ep):
            r = by_ep[e]
            w.writerow({
                "epoch": e,
                "train_loss": f"{r['train_loss']:.6f}" if r.get("train_loss") is not None else "",
                "train_acc": "",
                "train_lr": "",
                "val_loss": f"{r['val_loss']:.6f}" if r.get("val_loss") is not None else "",
                "val_acc": f"{r['val_acc']:.4f}" if r.get("val_acc") is not None else "",
            })
    print(f"Built metrics from checkpoints: {out_csv} ({len(by_ep)} points)")
    return out_csv


def per_class_scores(cm):
    n = cm.shape[0]
    recall = np.zeros(n)
    precision = np.zeros(n)
    for i in range(n):
        row_sum = cm[i, :].sum()
        col_sum = cm[:, i].sum()
        recall[i] = cm[i, i] / row_sum if row_sum else 0.0
        precision[i] = cm[i, i] / col_sum if col_sum else 0.0
    return recall, precision


def plot_yolo_panel(ax, x, y, title, ylim=None, xlabel=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ax.plot(x, y, marker=".", linewidth=1.5, markersize=8, label="results")
    if len(y) >= 2:
        ax.plot(x, smooth(y), ":", linewidth=2, color="#FF7F0E", label="smooth")
    ax.set_title(title, fontsize=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlabel:
        ax.set_xlabel("epoch")
    ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.5)


def plot_training_dashboard(epoch_data, iter_data, cm, class_names, save_path):
    fig, ax = plt.subplots(2, 5, figsize=(9, 5.5), tight_layout=True)
    ax = ax.ravel()
    max_ep = int(epoch_data["epoch"].max())
    xticks = [0, max(max_ep // 2, 1), max_ep] if max_ep > 0 else [0, 50, 100]

    if "train_loss" in epoch_data:
        plot_yolo_panel(ax[0], *epoch_data["train_loss"], "train/loss")
    if "train_acc" in epoch_data:
        plot_yolo_panel(ax[1], *epoch_data["train_acc"], "train/acc", ylim=(0, 102))
    if "train_lr" in epoch_data:
        plot_yolo_panel(ax[2], *epoch_data["train_lr"], "train/lr", xlabel=True)
    if "train_acc" in epoch_data:
        plot_yolo_panel(ax[3], *epoch_data["train_acc"], "metrics/accuracy", ylim=(0, 102))
    if "train_loss_iter" in iter_data:
        x, y = iter_data["train_loss_iter"]
        ax[4].plot(x, y, marker=".", linewidth=1.5, markersize=6, label="results")
        if len(y) >= 2:
            ax[4].plot(x, smooth(y), ":", linewidth=2, color="#FF7F0E", label="smooth")
        ax[4].set_title("train/loss (iter)", fontsize=9)
        ax[4].set_xlabel("step")
        ax[4].grid(True, linestyle="--", linewidth=0.35, alpha=0.5)

    if "val_loss" in epoch_data:
        plot_yolo_panel(ax[5], *epoch_data["val_loss"], "val/loss")
    if "val_acc" in epoch_data:
        plot_yolo_panel(ax[6], *epoch_data["val_acc"], "val/acc", ylim=(0, 102))
        plot_yolo_panel(ax[7], *epoch_data["val_acc"], "metrics/accuracy", ylim=(0, 102))

    recall, precision = per_class_scores(cm)
    y_pos = np.arange(len(class_names))
    ax[8].barh(y_pos, recall * 100, color="#1F77B4", height=0.7)
    ax[8].set_yticks(y_pos)
    ax[8].set_yticklabels(class_names, fontsize=7)
    ax[8].set_xlim(0, 105)
    ax[8].set_title("metrics/recall(C)", fontsize=9)
    ax[8].grid(True, axis="x", linestyle="--", alpha=0.5)

    ax[9].barh(y_pos, precision * 100, color="#2CA02C", height=0.7)
    ax[9].set_yticks(y_pos)
    ax[9].set_yticklabels(class_names, fontsize=7)
    ax[9].set_xlim(0, 105)
    ax[9].set_title("metrics/precision(C)", fontsize=9)
    ax[9].grid(True, axis="x", linestyle="--", alpha=0.5)

    for i in range(8):
        ax[i].set_xlim(0, max_ep + 1)
        ax[i].xaxis.set_major_locator(ticker.FixedLocator(xticks))
        ax[i].tick_params(labelsize=7.5)
    ax[0].legend(fontsize=7, loc="upper right", framealpha=0.85)

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved training dashboard: {save_path}")
