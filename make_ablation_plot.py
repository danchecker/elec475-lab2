"""
ELEC 475: Lab 2 - Multi-curve ablation summary figure built from saved train.py logs.

Reads the epoch tables that train.py prints (saved with `| tee` into the .txt files) and writes
plots/ablation_summary.png. Run from the project root:

    python make_ablation_plot.py --log-dir /content/drive/MyDrive/lab2_out --out plots/ablation_summary.png
"""
import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# epoch | train loss | train acc | val loss | val acc | val F1 | time
ROW = re.compile(r"^\s*(\d+)\s*\|" + r"\s*([\d.]+)\s*\|" * 5 + r"\s*([\d.]+)\s*\|")

NORM = {"No normalization": "alexnet_none.txt", "LRN": "alexnet_lrn.txt", "BatchNorm": "alexnet_bn.txt"}
AUG = {"No augmentation": "alexnet_no_aug.txt", "With augmentation": "alexnet_with_aug.txt"}
BARS = [
    ("None", "alexnet_none.txt"), ("LRN", "alexnet_lrn.txt"), ("BN (baseline)", "alexnet_bn.txt"),
    ("BN lr 1e-4", "abl_bn_lr1e-4.txt"), ("BN lr 1e-2", "abl_bn_lr1e-2.txt"),
    ("None lr 1e-2", "abl_none_lr1e-2.txt"), ("BN batch 32", "abl_bn_bs32.txt"),
    ("BN batch 512", "abl_bn_bs512.txt"),
]


def parse(path):
    """Return a list of (epoch, train_loss, train_acc, val_loss, val_acc, val_f1, time) rows."""
    rows = []
    if not os.path.exists(path):
        print(f"  missing: {path}")
        return rows
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = ROW.match(line)
            if m:
                rows.append(tuple(float(x) for x in m.groups()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--out", default="plots/ablation_summary.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) normalization study: validation accuracy
    for label, fn in NORM.items():
        r = parse(os.path.join(args.log_dir, fn))
        if r:
            ax[0, 0].plot([x[0] for x in r], [x[4] for x in r], marker="o", ms=3, label=label)
    ax[0, 0].set(title="Normalization study: validation accuracy", xlabel="Epoch", ylabel="Val accuracy (%)")
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)

    # (b) normalization study: training loss
    for label, fn in NORM.items():
        r = parse(os.path.join(args.log_dir, fn))
        if r:
            ax[0, 1].plot([x[0] for x in r], [x[1] for x in r], marker="o", ms=3, label=label)
    ax[0, 1].set(title="Normalization study: training loss", xlabel="Epoch", ylabel="Train loss")
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)

    # (c) augmentation study: train vs validation accuracy
    for (label, fn), color in zip(AUG.items(), ["tab:red", "tab:green"]):
        r = parse(os.path.join(args.log_dir, fn))
        if r:
            ep = [x[0] for x in r]
            ax[1, 0].plot(ep, [x[2] for x in r], "--", color=color, label=f"{label}: train")
            ax[1, 0].plot(ep, [x[4] for x in r], "-", color=color, label=f"{label}: validation")
    ax[1, 0].set(title="Augmentation study (BN, 15 epochs): generalization gap", xlabel="Epoch", ylabel="Accuracy (%)")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    # (d) best validation accuracy across the 10-epoch AlexNet runs
    labels, best = [], []
    for label, fn in BARS:
        r = parse(os.path.join(args.log_dir, fn))
        if r:
            labels.append(label)
            best.append(max(x[4] for x in r))
    bars = ax[1, 1].barh(labels, best, color="tab:blue")
    ax[1, 1].invert_yaxis()
    ax[1, 1].bar_label(bars, fmt="%.1f", padding=3)
    ax[1, 1].set(title="Best validation accuracy, AlexNet-64, 10 epochs", xlabel="Val accuracy (%)", xlim=(0, 100))
    ax[1, 1].grid(alpha=0.3, axis="x")

    fig.suptitle("ELEC 475 Lab 2: AlexNet-64 ablation summary (CIFAR-10, seed 42)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
    for label, b in zip(labels, best):
        print(f"  {label:16s} best val acc {b:.2f}%")


if __name__ == "__main__":
    main()
