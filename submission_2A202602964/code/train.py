"""train.py — Vòng lặp huấn luyện, đánh giá, dự đoán và quản lý thí nghiệm.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).
"""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

# Cấu hình mặc định = BASELINE (M-base)
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base (SGD+momentum 0.9, He init)",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.05,                   # Chọn bằng val
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(np.mean(f1))


@torch.no_grad()
def predict(model: torch.nn.Module, X: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1))
    return torch.cat(preds, dim=0)


@torch.no_grad()
def evaluate(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor,
             loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    n_samples = len(X)

    for i in range(0, n_samples, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        if loss_name == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
        elif loss_name == "mse":
            y_onehot = F.one_hot(yb, num_classes=7).float()
            loss = F.mse_loss(logits, y_onehot, reduction="sum")
        else:
            raise ValueError(f"Loss không hỗ trợ: {loss_name}")

        total_loss += loss.item()
        all_preds.append(torch.argmax(logits, dim=1))

    pred_cat = torch.cat(all_preds, dim=0).cpu().numpy()
    y_np = y.cpu().numpy()

    loss_avg = total_loss / n_samples
    acc = float(np.mean(pred_cat == y_np))

    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (y_np, pred_cat), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": loss_avg, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits: torch.Tensor, y: torch.Tensor, loss_name: str) -> torch.Tensor:
    """"ce"  : cross-entropy nhận logit thô và nhãn int64.
       "mse" : MSE giữa logit và one-hot của y.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=7).float()
        return F.mse_loss(logits, y_onehot)
    else:
        raise ValueError(f"Hàm loss không hợp lệ: {loss_name}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    seed = cfg.get("seed", 1)
    set_seed(seed)

    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init_type = cfg.get("init", "he")
    loss_name = cfg.get("loss", "ce")
    opt_name = cfg.get("optimizer", "sgd_momentum")
    lr = float(cfg.get("lr", 0.05))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    batch_size = int(cfg.get("batch", 512))
    epochs = int(cfg.get("epochs", 20))
    clip_norm = cfg.get("clip_norm", None)
    precision = cfg.get("precision", "fp32")

    device = data["X_tr"].device
    is_cuda = (device.type == "cuda")
    device_type = "cuda" if is_cuda else "cpu"

    # Kiểm tra kiến trúc và số tham số
    model = MLP(hidden=hidden, dropout=dropout, init=init_type).to(device)
    actual_params = count_params(model)
    expected = EXPECTED_PARAMS.get(hidden)
    if expected is not None:
        assert actual_params == expected, f"Số tham số ({actual_params}) lệch so với quy định ({expected})"

    optimizer = build_optimizer(
        opt_name, model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
    )

    # Mixed precision setup
    use_amp = (precision == "bf16" or (precision == "fp16" and is_cuda))
    amp_dtype = torch.float16 if (precision == "fp16" and is_cuda) else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and precision == "fp16" and is_cuda) else None

    # Đo loss bước 0 trên val trước khi cập nhật
    step0_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
    step0_loss = float(step0_eval["loss"])

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    diverged = False

    # Máy phát sinh ngẫu nhiên cho shuffling lô
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Chọn tập con train cố định để tính train loss sau mỗi epoch (50,000 mẫu đầu để tiết kiệm thời gian)
    train_subset_X = data["X_tr"][:50000]
    train_subset_y = data["y_tr"][:50000]

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        grad_norms_epoch = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator=gen, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True
                break

            if scaler is not None:
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                grad_norms_epoch.append(gn)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                grad_norms_epoch.append(gn)
                optimizer.step()

        if is_cuda:
            torch.cuda.synchronize(device)

        epoch_time = time.time() - epoch_t0

        if diverged:
            print(f"[{cfg.get('exp_id')}] Epoch {epoch}: Loss bị NaN/inf! Dừng sớm.")
            break

        # Đánh giá cuối epoch ở chế độ eval
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
        train_eval = evaluate(model, train_subset_X, train_subset_y, loss_name=loss_name)

        mean_gn = float(np.mean(grad_norms_epoch)) if grad_norms_epoch else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        # Cập nhật checkpoint val loss tốt nhất
        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if is_cuda:
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    else:
        peak_mem_MB = 0.0

    avg_epoch_time = float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0

    if best_state is None and not diverged:
        best_state = copy.deepcopy(model.state_dict())

    # Chỉ số tại best epoch
    best_idx = best_epoch - 1 if history["epoch"] else 0
    val_acc_at_best = history["val_acc"][best_idx] if history["val_acc"] else 0.0
    val_f1_at_best = history["val_macro_f1"][best_idx] if history["val_macro_f1"] else 0.0
    final_tr_loss = history["train_loss"][-1] if history["train_loss"] else float("nan")
    final_v_loss = history["val_loss"][-1] if history["val_loss"] else float("nan")

    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss if not diverged else float("nan"),
        "best_epoch": best_epoch,
        "final_train_loss": final_tr_loss,
        "final_val_loss": final_v_loss,
        "val_acc": val_acc_at_best,
        "val_macro_f1": val_f1_at_best,
        "time_per_epoch_s": avg_epoch_time,
        "peak_mem_MB": peak_mem_MB,
        "diverged": diverged,
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id: np.ndarray, preds: np.ndarray, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    df = pd.DataFrame({"row_id": row_id.astype(int), "pred": preds.astype(int)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Đã ghi dự đoán eval vào {path} ({len(df):,d} dòng)")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> np.ndarray:
    """Nạp best_state, dự đoán trên toàn bộ tập eval, và ghi predictions_eval.csv."""
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init_type = cfg.get("init", "he")
    device = data["X_eval"].device

    model = MLP(hidden=hidden, dropout=dropout, init=init_type).to(device)
    model.load_state_dict(result["best_state"])

    preds = predict(model, data["X_eval"]).cpu().numpy()
    write_predictions(data["eval_row_id"], preds, pred_path)
    return preds
