"""Đọc/kiểm tra dữ liệu Cleveland và chia tập có phân tầng."""

import csv
import math
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from .config import COLUMNS, Settings

if TYPE_CHECKING:
    import pandas as pd


def inspect_raw_csv(path: Path) -> dict:
    """Kiểm tra CSV không header bằng stdlib; chưa impute hoặc scale.

    Chấp nhận '?' hoặc ô trống ở đầu vào, nhưng không chấp nhận nhãn thiếu.
    Trả về số dòng/cột, nhãn gốc, nhãn nhị phân và số ô thiếu theo cột.
    """
    missing = Counter({name: 0 for name in COLUMNS})
    labels = Counter()
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.reader(stream), start=1):
            if len(row) != len(COLUMNS):
                raise ValueError(f"Dòng {line}: cần {len(COLUMNS)} cột, nhận {len(row)}.")
            for name, value in zip(COLUMNS, row):
                value = value.strip()
                if value in {"", "?"}:
                    if name == "target":
                        raise ValueError(f"Dòng {line}: nhãn target bị thiếu.")
                    missing[name] += 1
                    continue
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError(f"Dòng {line}: {name} không hữu hạn.")
                if name == "target":
                    if number not in (0, 1, 2, 3, 4):
                        raise ValueError(f"Dòng {line}: target phải thuộc 0..4.")
                    labels[int(number)] += 1
            count += 1
    if not count:
        raise ValueError("CSV rỗng.")
    return {
        "rows": count,
        "columns": len(COLUMNS),
        "original_target_counts": dict(sorted(labels.items())),
        "binary_target_counts": {0: labels[0], 1: sum(v for k, v in labels.items() if k > 0)},
        "missing_by_column": dict(missing),
    }


def load_cleveland(path: Path) -> "pd.DataFrame":
    """Đã triển khai: 14 cột đúng schema; '?' -> NaN; target 0..4 -> 0/1.

    Kiểm tra nhãn TRƯỚC khi nhị phân hóa để tránh biến nhãn lỗi thành lớp 0.
    """
    import pandas as pd

    inspect_raw_csv(path)
    frame = pd.read_csv(
        path, header=None, names=COLUMNS, na_values=["?", ""], skipinitialspace=True
    )
    for name in COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["target"] = (frame["target"] > 0).astype(int)
    return frame


def split_dataset(frame: "pd.DataFrame", settings: Settings) -> dict:
    """Trả {'train': (X,y), 'val': (X,y), 'test': (X,y)}.

    Dùng train_test_split 2 lần, stratify=y, random_state=settings.seed.
    test_size lần 1 = val_ratio + test_ratio; lần 2 = test_ratio/(val+test).
    Giữ index gốc để kiểm tra không trùng mẫu. Không impute trước khi chia.
    Với dữ liệu hiện tại và 80:10:10, mong đợi 242/30/31 mẫu.
    """
    from sklearn.model_selection import train_test_split

    X, y = frame.drop(columns=[settings.target]), frame[settings.target]
    holdout_ratio = settings.val_ratio + settings.test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=holdout_ratio,
        stratify=y,
        random_state=settings.seed,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=settings.test_ratio / holdout_ratio,
        stratify=y_temp,
        random_state=settings.seed,
    )
    return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
