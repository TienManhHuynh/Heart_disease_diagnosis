# Bàn giao Project 3.1

## Mở nhanh

1. Đọc reports/final_report.md hoặc output/pdf/Heart_Disease_Project_Report.pdf.
2. Chạy `.venv/Scripts/python.exe -m streamlit run app.py` để xem demo.
3. Chạy `.venv/Scripts/python.exe scripts/check_project.py` để kiểm tra source và unit test.
4. Mở bốn notebook theo thứ tự; code và nhận xét đã được hoàn thiện.

## Cài trên máy mới

Giải nén ZIP, mở terminal tại thư mục có pyproject.toml, dùng Python 3.11 trên Windows:
`python -m venv .venv`, sau đó `.venv/Scripts/python.exe -m pip install -r requirements-lock.txt`.
Không sao chép môi trường .venv từ máy cũ. Artifact hiện tại được tạo với phiên bản trong lock/metadata.

## Chạy lại và quy tắc lựa chọn

`python -m heart_disease run` tạo run chính mới. Sau đó chạy `extensions`, `report`, rồi xuất PDF và đóng gói.
Cấu hình mặc định/report phân tích hiện hành dành cho Cleveland local và seed 42. Nếu đổi dữ liệu, seed hoặc mô hình,
phải đọc lại toàn bộ nhận xét trong báo cáo để đối chiếu các ví dụ và kết luận, không giữ diễn giải cũ một cách máy móc.

Run gốc của phần chính: 20261102T164927743944Z. Selected model: random_forest/original, 50 cây, max_depth=5.
KMeans/stacking là phần mở rộng thêm sau khi đã xem test; không dùng để chọn lại mô hình demo.

## Khi trình bày hoặc bảo vệ

- Giải thích vì sao target 1-4 gộp thành 1, vì sao categorical dù lưu bằng số.
- Chỉ ra vị trí fit/transform và cách pipeline tránh rò rỉ giữa CV fold.
- Phân biệt FE và selection; đọc danh sách MI/DT và tác động không nhất quán.
- Trình bày confusion matrix: TN=15, FP=2, FN=1, TP=13 trên 31 mẫu.
- Giải thích SHAP cùng thang xác suất; không diễn giải thành nguyên nhân bệnh.
- Nêu lưới nhỏ, test nhỏ, giới hạn repeated CV đã chọn hyperparameter trên cùng train.

## Các file quan trọng

- reports/executed_notebooks/: notebook đã chạy, có kết quả để đọc ngay.
- reports/delivery_verification.json: bằng chứng kiểm tra tự động.
- reports/final_report.md: báo cáo phân tích chính.
- reports/results.csv: 28 cấu hình chính; reports/extensions/<run>/results.csv: 8 cấu hình bổ sung.
- reports/error_analysis.csv và test_predictions.csv: lỗi và xác suất từng hàng.
- reports/demo_examples.csv: ba ví dụ thật, đủ 13 giá trị đầu vào.
- reports/shap_local_contributions.csv: dấu/độ lớn đóng góp cho ví dụ local.
- artifacts/metadata: split, version, hash và lựa chọn; artifacts/models: pipeline đầy đủ.
- output/delivery: bản ZIP bàn giao, có checksum và manifest.

## Minh họa demo

Ứng dụng có kiểm tra AppTest và bảng ví dụ đầu vào/đầu ra thật trong reports/demo_examples.csv.
Có thể chạy demo local để minh họa trực tiếp khi trình bày.
