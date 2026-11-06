"""Xuất báo cáo Markdown từ kết quả thực nghiệm thật."""

from pathlib import Path


def _table(frame, columns):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        cells = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_run_report(manifest, table, evaluation, destination):
    model, variant = manifest["selected_key"].split("/")
    row = table[(table.model == model) & (table.variant == variant)].iloc[0]
    stability = evaluation["stability_training_only"]
    split_sizes = "/".join(str(len(manifest["split_indices"][k])) for k in ("train", "val", "test"))
    text = f"""# Kết quả sơ bộ — Heart Disease Project 3.1

Báo cáo sinh từ lần chạy `{manifest["run_id"]}`, không lấy số liệu từ PDF.

## Thiết lập

- Cleveland local: 303 mẫu, 13 feature; target nhị phân.
- SHA-256: `{manifest["raw_sha256"]}`.
- Seed: {manifest["settings"]["seed"]}; train/val/test: {split_sizes} mẫu.
- CV: Stratified {manifest["settings"]["cv_folds"]}-fold, scoring `{manifest["settings"]["cv_scoring"]}`.
- Mọi preprocessing/selection được fit lại trong từng training fold.
- Quy tắc chọn: {manifest["selection_rule"]}
- Lưới nhỏ phục vụ lần chạy sơ bộ; xem models.parameter_grid và cv_results.csv.
- Estimator cuối fit trên train; không refit trên validation/test.

## Cấu hình được chọn trước khi xem test

`{manifest["selected_key"]}`; best params: `{row["best_params"]}`.
CV score: {row["cv_mean"]:.4f} ± {row["cv_std"]:.4f}.
Validation accuracy/F1: {row["val_accuracy"]:.4f}/{row["val_f1"]:.4f}.
Test accuracy/F1/recall lớp 1: {row["test_accuracy"]:.4f}/{row["test_f1"]:.4f}/{row["test_recall"]:.4f}.
Không dùng bảng test để thay đổi lựa chọn.

## Bảng đối chiếu

{_table(table, ["model", "variant", "cv_mean", "cv_std", "val_accuracy", "val_f1", "test_accuracy", "test_f1", "test_recall"])}

## Độ ổn định

Repeated CV trên TRAIN của cấu hình đã chọn: {stability["n_splits"]} fold × {stability["n_repeats"]} lần;
F1 trung bình {stability["mean"]:.4f}, độ lệch chuẩn {stability["std"]:.4f}.
Hyperparameter đã được chọn trên cùng train: đây là phân tích mô tả, không phải nested CV hoặc ước lượng độc lập.

## Hạn chế và bước tiếp theo

Test chỉ có {evaluation["test_samples"]} mẫu; một mẫu làm accuracy đổi {100 / evaluation["test_samples"]:.2f} điểm phần trăm.
Dữ liệu nhỏ và lịch sử; demo phục vụ học tập, chưa có kiểm định ngoài Cleveland.
Báo cáo phân tích đầy đủ được tạo bằng lệnh report sau extensions.
KMeans/stacking được đánh giá riêng; không đổi seed để chọn điểm đẹp.

## Tái lập

Chạy `.venv/Scripts/python.exe -m heart_disease run` tại thư mục project.
Phiên bản dependencies trong requirements-lock.txt và metadata artifact.
Classification report, confusion matrix và SHAP: `reports/runs/{manifest["run_id"]}/`.
"""
    Path(destination).write_text(text, encoding="utf-8")
    return Path(destination)
