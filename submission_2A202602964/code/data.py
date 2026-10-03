"""data.py — Chuẩn bị và tiền xử lý dữ liệu CoverType.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.
Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    p = Path(processed_dir)
    train_path = p / "train.npz"
    eval_path = p / "eval.npz"
    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file npz trong {processed_dir}. Hãy chạy split_data.py trước!")

    with np.load(train_path) as tr:
        X_train_full = tr["X"].astype(np.float32)
        y_train_full = tr["y"].astype(np.int64)

    with np.load(eval_path) as ev:
        X_eval = ev["X"].astype(np.float32)
        y_eval = ev["y"].astype(np.int64)
        eval_row_id = ev["row_id"].astype(np.int64)

    assert X_train_full.ndim == 2 and X_train_full.shape[1] == 54, f"X_train shape không đúng: {X_train_full.shape}"
    assert y_train_full.ndim == 1 and len(y_train_full) == len(X_train_full), "y_train shape không đúng"
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, f"X_eval shape không đúng: {X_eval.shape}"
    assert y_eval.ndim == 1 and len(y_eval) == len(X_eval), "y_eval shape không đúng"
    assert len(eval_row_id) == len(X_eval), "eval_row_id shape không đúng"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Lý do: Không được tính trên toàn bộ dữ liệu hay eval để tránh rò rỉ thông tin (data leakage).
    """
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0)
    # Tránh chia cho 0 nếu có cột std = 0
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    # Đưa tensor lên device
    dev = torch.device(device)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=dev)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=dev)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=dev)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=dev)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=dev)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=dev)

    # Thống kê kiểm tra
    print(f"Dataset summary:")
    print(f"  Train : {len(X_tr_t):,d} mẫu (80%)")
    print(f"  Val   : {len(X_val_t):,d} mẫu (20%)")
    print(f"  Eval  : {len(X_eval_t):,d} mẫu")

    # Accuracy đoán lớp đa số trên val
    val_counts = np.bincount(y_val, minlength=7)
    majority_class = val_counts.argmax()
    majority_acc = val_counts[majority_class] / len(y_val)
    print(f"  Majority class on Val: class {majority_class} (acc = {majority_acc:.4f})")

    # Kiểm tra chuẩn hoá trên X_tr
    mean_check = X_tr[:, :N_NUMERIC].mean(axis=0)
    std_check = X_tr[:, :N_NUMERIC].std(axis=0)
    print(f"  Standardized train features: mean max={np.max(np.abs(mean_check)):.2e}, std mean={np.mean(std_check):.4f}")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "majority_class": majority_class,
        "majority_acc": majority_acc,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
