# Kết quả sơ bộ — Heart Disease Project 3.1

Báo cáo sinh từ lần chạy `20261102T164927743944Z`, không lấy số liệu từ PDF.

## Thiết lập

- Cleveland local: 303 mẫu, 13 feature; target nhị phân.
- SHA-256: `70a023033aaf840596737ad250eea055762ad825036ac73b5b29168dc638c7c2`.
- Seed: 42; train/val/test: 242/30/31 mẫu.
- CV: Stratified 5-fold, scoring `accuracy`.
- Mọi preprocessing/selection được fit lại trong từng training fold.
- Quy tắc chọn: Highest train CV mean; ties by validation F1, then model and variant names.
- Lưới nhỏ phục vụ lần chạy sơ bộ; xem models.parameter_grid và cv_results.csv.
- Estimator cuối fit trên train; không refit trên validation/test.

## Cấu hình được chọn trước khi xem test

`random_forest/original`; best params: `{'model__n_estimators': 50}`.
CV score: 0.8263 ± 0.0181.
Validation accuracy/F1: 0.9667/0.9655.
Test accuracy/F1/recall lớp 1: 0.9032/0.8966/0.9286.
Không dùng bảng test để thay đổi lựa chọn.

## Bảng đối chiếu

| model | variant | cv_mean | cv_std | val_accuracy | val_f1 | test_accuracy | test_f1 | test_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_bayes | original | 0.8181 | 0.0307 | 0.9000 | 0.9032 | 0.8387 | 0.8387 | 0.9286 |
| naive_bayes | fe | 0.8137 | 0.0407 | 0.9000 | 0.9032 | 0.8387 | 0.8387 | 0.9286 |
| naive_bayes | original_dt | 0.8224 | 0.0198 | 0.9333 | 0.9333 | 0.8387 | 0.8387 | 0.9286 |
| naive_bayes | fe_dt | 0.8140 | 0.0185 | 0.8667 | 0.8667 | 0.8710 | 0.8667 | 0.9286 |
| knn | original | 0.7854 | 0.0289 | 0.9000 | 0.8966 | 0.8710 | 0.8571 | 0.8571 |
| knn | fe | 0.8014 | 0.0435 | 0.8667 | 0.8667 | 0.8710 | 0.8667 | 0.9286 |
| knn | original_dt | 0.7682 | 0.0496 | 0.9333 | 0.9286 | 0.8387 | 0.8387 | 0.9286 |
| knn | fe_dt | 0.8017 | 0.0205 | 0.9000 | 0.8966 | 0.8065 | 0.8000 | 0.8571 |
| decision_tree | original | 0.7728 | 0.0254 | 0.9333 | 0.9286 | 0.8065 | 0.7857 | 0.7857 |
| decision_tree | fe | 0.7645 | 0.0246 | 0.9333 | 0.9286 | 0.8065 | 0.7857 | 0.7857 |
| decision_tree | original_dt | 0.7769 | 0.0200 | 0.9333 | 0.9286 | 0.8065 | 0.7857 | 0.7857 |
| decision_tree | fe_dt | 0.7645 | 0.0246 | 0.9333 | 0.9286 | 0.8065 | 0.7857 | 0.7857 |
| random_forest | original | 0.8263 | 0.0181 | 0.9667 | 0.9655 | 0.9032 | 0.8966 | 0.9286 |
| random_forest | fe | 0.8222 | 0.0313 | 0.9667 | 0.9655 | 0.8710 | 0.8571 | 0.8571 |
| random_forest | original_dt | 0.8223 | 0.0211 | 0.9333 | 0.9286 | 0.8710 | 0.8571 | 0.8571 |
| random_forest | fe_dt | 0.8058 | 0.0100 | 0.9000 | 0.8966 | 0.8387 | 0.8148 | 0.7857 |
| adaboost | original | 0.7891 | 0.0483 | 0.9000 | 0.9032 | 0.8710 | 0.8667 | 0.9286 |
| adaboost | fe | 0.7810 | 0.0410 | 0.9667 | 0.9655 | 0.8065 | 0.8000 | 0.8571 |
| adaboost | original_dt | 0.7809 | 0.0488 | 0.9667 | 0.9655 | 0.8387 | 0.8387 | 0.9286 |
| adaboost | fe_dt | 0.7768 | 0.0485 | 0.9667 | 0.9655 | 0.8065 | 0.8000 | 0.8571 |
| gradient_boosting | original | 0.7891 | 0.0312 | 0.9333 | 0.9333 | 0.8710 | 0.8571 | 0.8571 |
| gradient_boosting | fe | 0.8139 | 0.0143 | 0.9333 | 0.9333 | 0.8710 | 0.8571 | 0.8571 |
| gradient_boosting | original_dt | 0.7727 | 0.0294 | 0.9333 | 0.9333 | 0.8710 | 0.8571 | 0.8571 |
| gradient_boosting | fe_dt | 0.8015 | 0.0220 | 0.9333 | 0.9333 | 0.8065 | 0.7857 | 0.7857 |
| xgboost | original | 0.8099 | 0.0093 | 0.9000 | 0.9032 | 0.8065 | 0.7857 | 0.7857 |
| xgboost | fe | 0.8056 | 0.0292 | 0.9000 | 0.8966 | 0.8387 | 0.8148 | 0.7857 |
| xgboost | original_dt | 0.8057 | 0.0175 | 0.9667 | 0.9655 | 0.8065 | 0.7857 | 0.7857 |
| xgboost | fe_dt | 0.7933 | 0.0145 | 0.9000 | 0.8966 | 0.8387 | 0.8148 | 0.7857 |

## Độ ổn định

Repeated CV trên TRAIN của cấu hình đã chọn: 5 fold × 3 lần;
F1 trung bình 0.7855, độ lệch chuẩn 0.0589.
Hyperparameter đã được chọn trên cùng train: đây là phân tích mô tả, không phải nested CV hoặc ước lượng độc lập.

## Hạn chế và bước tiếp theo

Test chỉ có 31 mẫu; một mẫu làm accuracy đổi 3.23 điểm phần trăm.
Dữ liệu nhỏ và lịch sử; demo phục vụ học tập, chưa có kiểm định ngoài Cleveland.
Đã bổ sung phân tích EDA/FE/sai số/SHAP trong final_report.md.
KMeans/stacking đã được triển khai và báo cáo riêng; không đổi seed để chọn điểm đẹp.

## Tái lập

Chạy `.venv/Scripts/python.exe -m heart_disease run` tại thư mục project.
Phiên bản dependencies trong requirements-lock.txt và metadata artifact.
Classification report, confusion matrix và SHAP: `reports/runs/20261102T164927743944Z/`.
