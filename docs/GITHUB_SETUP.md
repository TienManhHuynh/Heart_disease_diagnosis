# Đưa project lên GitHub

## Tên và mô tả repository

- **Tên đề xuất:** `heart-disease-diagnosis-ml`.
- **Description:** `Leakage-aware Cleveland Heart Disease ML benchmark with feature engineering, TreeSHAP explanations, and a Streamlit demo.`
- **Topics:** `machine-learning`, `heart-disease`, `scikit-learn`, `xgboost`, `shap`, `streamlit`, `classification`.

README.md là bản giới thiệu duy nhất của project, viết bằng tiếng Việt.
Các đường dẫn trong README đều tương đối để hoạt động khi clone sang máy khác.

## Cách 1: Git

Tạo repository trống trên tài khoản GitHub của bạn, rồi chạy tại thư mục project:

```bash
git init -b main
git add .
git status --short
git commit -m "Initial commit: heart disease ML project"
```

Ở trang repo vừa tạo, sao chép URL HTTPS. Thay `<REPOSITORY_URL>` bên dưới bằng URL đó:

```bash
git remote add origin <REPOSITORY_URL>
git push -u origin main
```

Trước commit, kiểm tra danh sách staged chỉ gồm source, notebook sạch, dữ liệu raw,
tài liệu, workflow và kết quả tham chiếu. `.gitignore` loại môi trường, cache, model,
file tạm và output. Gói CI sẽ chạy sau khi workflow được GitHub kích hoạt; kết quả local
không được trình bày như kết quả CI đã chạy trên GitHub.

## Cách 2: upload source qua trình duyệt

Chạy `python scripts/package_source.py`, giải nén ZIP trong `output/github/`,
rồi upload nội dung source đã giải nén. Gói này đã loại file runtime/local;
`.gitignore` không tự lọc khi bạn kéo thả cả thư mục trực tiếp qua trình duyệt.
Để bảo toàn thư mục ẩn `.github`, ưu tiên dùng Git ở cách 1.

## Bản source và bản bàn giao

- Bản source có `data/raw/cleveland.csv`, không cần tải dữ liệu khi chạy.
- Models/metadata được train lại sau khi clone; demo cần `benchmark` hoặc `run` trước.
- `reports/` giữ bảng và biểu đồ tham chiếu để đọc trực tiếp trên GitHub.
- `output/delivery/` là gói bàn giao đầy đủ local; không commit ZIP này vào source.
- File Colab cũ được lưu ở `.local/legacy/`, không đưa lên GitHub.

Dataset attribution đã có trong NOTICE.md. Mã nguồn chưa được gán một giấy phép
riêng; chọn license cho code theo quyền sử dụng bạn muốn trước khi cho phép tái sử dụng rộng rãi.
