"""plots.py — Vẽ biểu đồ thí nghiệm đơn lẻ và biểu đồ so sánh nhóm.

Mỗi thí nghiệm có một ảnh figures/<exp_id>.png gồm ít nhất 3 ô:
  (1) train_loss và val_loss theo epoch
  (2) val_acc và val_macro_f1 theo epoch
  (3) grad_norm theo epoch (đo trước khi clip)
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô."""
    cfg = result["cfg"]
    hist = result["history"]
    summary = result["summary"]
    exp_id = cfg.get("exp_id", "exp")

    epochs = hist.get("epoch", [])
    if not epochs:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    best_epoch = summary.get("best_epoch", 1)

    # Ô 1: Train & Val Loss
    ax1 = axes[0]
    ax1.plot(epochs, hist["train_loss"], label="Train Loss (eval mode)", color="#1f77b4", lw=2)
    ax1.plot(epochs, hist["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax1.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax1.set_title("Loss vs Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Ô 2: Val Accuracy & Macro-F1
    ax2 = axes[1]
    ax2.plot(epochs, hist["val_acc"], label="Val Acc", color="#2ca02c", lw=2)
    ax2.plot(epochs, hist["val_macro_f1"], label="Val Macro-F1", color="#d62728", lw=2)
    ax2.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax2.axhline(0.4876, color="gray", linestyle=":", alpha=0.7, label="Majority guess (~0.488)")
    ax2.set_title("Val Metrics vs Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim([0.0, 1.0])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # Ô 3: Gradient Norm (trước khi clip)
    ax3 = axes[2]
    ax3.plot(epochs, hist["grad_norm"], label="Grad Norm (L2)", color="#9467bd", lw=2)
    ax3.axvline(best_epoch, color="red", linestyle="--", alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax3.set_title("Grad Norm (pre-clip) vs Epoch", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("L2 Norm")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    # Tiêu đề tổng quát
    opt_str = f"{cfg.get('optimizer')} (lr={cfg.get('lr')})"
    arch_str = f"Hidden={cfg.get('hidden')}, Init={cfg.get('init')}, Batch={cfg.get('batch')}"
    fig.suptitle(f"[{exp_id}] {cfg.get('description', '')}\n{opt_str} | {arch_str}", fontsize=13, y=1.03)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục."""
    if not results:
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    metric_labels = {
        "val_loss": "Val Loss",
        "train_loss": "Train Loss",
        "val_acc": "Val Accuracy",
        "val_macro_f1": "Val Macro-F1",
        "grad_norm": "Grad Norm (L2)",
    }
    label_y = metric_labels.get(metric, metric)

    for res in results:
        cfg = res["cfg"]
        hist = res["history"]
        exp_id = cfg.get("exp_id", "exp")
        epochs = hist.get("epoch", [])
        vals = hist.get(metric, [])
        if epochs and vals:
            ax.plot(epochs, vals, marker="o", markersize=3, label=f"{exp_id} ({cfg.get('description', '')})")

    ax.set_title(title or f"Comparison: {label_y}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(label_y, fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
