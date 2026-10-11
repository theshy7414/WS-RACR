#!/usr/bin/env python3
"""
手勢辨識：驗證集評估 + YOLO 風格 2×5 訓練儀表板 + 混淆矩陣。

  python eval_val_plot.py \\
      --checkpoint hand_gesture_model_acc100.00_20251204_033943.pth \\
      --data_dir datasets \\
      --metrics checkpoints/metrics.csv
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from ws_racr.perception.gesture.pose_dataset import create_datasets
from ws_racr.perception.gesture.pose_model import HandGestureClassifier

from yolo_plots import (
    build_metrics_from_checkpoints,
    load_epoch_metrics,
    plot_training_dashboard,
)


def evaluate_val(model, val_loader, device, num_classes):
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            preds = model(features).argmax(dim=1)
            for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                cm[t, p] += 1

    acc = 100.0 * np.trace(cm) / cm.sum() if cm.sum() else 0.0
    return cm, acc


def plot_confusion_matrix(cm, class_names, save_path):
    import seaborn as sns

    cm_display = cm.T.astype(float)
    annot = [[str(int(v)) if v > 0 else "" for v in row] for row in cm_display]
    vmax = max(float(cm_display.max()), 1.0)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        cm_display, ax=ax, annot=annot, fmt="",
        annot_kws={"size": 10}, cmap="Blues", norm=norm,
        linewidths=0.5, linecolor="white", square=True,
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={"shrink": 0.82},
    )
    ax.set_xlabel("True", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    total = int(cm.sum())
    acc = 100.0 * np.trace(cm) / total if total else 0
    ax.set_title(f"Confusion Matrix (val, n={total}, acc={acc:.1f}%)", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved CM: {save_path}  (sum={total}, acc={acc:.2f}%)")


def plot_dashboard_from_checkpoint(ckpt_state, class_names, cm, save_path):
    """無 metrics.csv 時，用 checkpoint 最後一輪指標畫簡化儀表板（單點）。"""
    ep = np.array([ckpt_state.get("epoch", 0) + 1], dtype=float)
    epoch_data = {
        "epoch": ep,
        "train_loss": (ep, np.array([ckpt_state.get("train_loss", 0.0)])),
        "val_loss": (ep, np.array([ckpt_state.get("val_loss", 0.0)])),
    }
    class_acc = ckpt_state.get("class_acc", {})
    if class_acc:
        train_acc = np.mean([class_acc.get(n, 0) for n in class_names])
        epoch_data["train_acc"] = (ep, np.array([train_acc]))
        epoch_data["val_acc"] = (ep, np.array([ckpt_state.get("val_acc", train_acc)]))
    plot_training_dashboard(epoch_data, {}, cm, class_names, save_path)


def main():
    parser = argparse.ArgumentParser(description="Real val eval + figures for gesture")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="datasets")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--tensorboard", default=None)
    parser.add_argument("--out_dir", default="../../paper/figures")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(
            f"ERROR: data_dir not found: {data_dir.resolve()}\n"
            "請將手勢 PNG 放在 src/gesture_detection/datasets/\n"
            "  idle_*.png, stop_*.png, receive_*.png, pointing_*.png"
        )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        sys.exit(f"ERROR: checkpoint not found: {ckpt_path.resolve()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    train_ds, val_ds = create_datasets(str(data_dir), args.train_ratio, args.seed)
    full = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds
    class_names = full.class_names

    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "model_state_dict" not in state:
        sys.exit("ERROR: checkpoint must contain model_state_dict")

    model = HandGestureClassifier(
        input_dim=state.get("input_dim", 68),
        num_classes=state.get("num_classes", len(class_names)),
    )
    model.load_state_dict(state["model_state_dict"])
    model.to(device)

    cm, acc = evaluate_val(model, val_loader, device, len(class_names))
    print(f"Val accuracy: {acc:.2f}%  (n={cm.sum()})")

    plot_confusion_matrix(cm, class_names, out_dir / "gesture_recog_confusion_matrix.png")

    metrics_path = Path(args.metrics) if args.metrics else ckpt_path.parent / "metrics.csv"
    dashboard = out_dir / "gesture_recog_training.png"

    if not metrics_path.is_file():
        built = build_metrics_from_checkpoints(ckpt_path.parent, metrics_path)
        if built:
            metrics_path = built

    if metrics_path.is_file():
        epoch_data = load_epoch_metrics(metrics_path)
        plot_training_dashboard(epoch_data, {}, cm, class_names, dashboard)
    else:
        print(f"WARN: no metrics; using checkpoint snapshot for dashboard")
        plot_dashboard_from_checkpoint(state, class_names, cm, dashboard)

    shutil.copy(dashboard, out_dir / "gesture_recog_loss_acc.png")
    print(f"Also copied -> {out_dir / 'gesture_recog_loss_acc.png'}")


if __name__ == "__main__":
    main()
