"""optimizer.py — Bộ tối ưu, scheduler và cắt gradient.

File này gom việc chọn bộ tối ưu và cắt gradient để train.py gọn và mọi thí nghiệm công bằng.
Công thức (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8) -> torch.optim.Optimizer:
    """Trả về một torch.optim.Optimizer tương ứng."""
    name_lower = name.lower()
    if name_lower not in OPTIMIZERS:
        raise ValueError(f"Optimizer '{name}' không hợp lệ. Chọn một trong {OPTIMIZERS}")

    if name_lower == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name_lower == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name_lower == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name_lower == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học."""
    if name is None or name.lower() in ("none", ""):
        return None
    name_lower = name.lower()
    if name_lower == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, **kwargs)
    elif name_lower == "step":
        step_size = kwargs.get("step_size", total_steps // 2)
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Scheduler không hỗ trợ: {name}")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Khi max_norm is None, ta truyền float('inf') vào clip_grad_norm_ để tính chuẩn mà không cắt.
    """
    # Lấy danh sách tham số có grad
    param_list = [p for p in params if p.grad is not None]
    if not param_list:
        return 0.0

    if max_norm is None or max_norm <= 0:
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(param_list, max_norm)

    return float(total_norm)
