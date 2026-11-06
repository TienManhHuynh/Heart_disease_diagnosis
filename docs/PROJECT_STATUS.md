# Checklist triển khai và bàn giao

Các phần thuộc phạm vi project đã được triển khai. Đọc reports/final_report.md và README để sử dụng.

- [x] Dữ liệu, EDA, phân tầng 242/30/31, kiểm tra missing/duplicate.
- [x] Original, FE, Original+DT, FE+DT; preprocessing/selection trong từng CV fold.
- [x] Bảy classifier chính x bốn biến thể; CV, validation và lựa chọn trước test.
- [x] KMeans mapping nhãn từ train và stacking với preprocessing bên trong OOF.
- [x] Tám cấu hình mở rộng lưu riêng, đánh dấu exploratory_post_hoc, giữ nguyên demo chính.
- [x] Metrics, confusion matrix, sai số từng mẫu, false negative/positive và độ ổn định.
- [x] TreeSHAP global/local, đóng góp cụ thể và kiểm chứng additivity.
- [x] Nhận xét cho cả bốn notebook và báo cáo Markdown đầy đủ, không còn ô TODO.
- [x] Báo cáo PDF có biểu đồ, nguồn và giới hạn phương pháp.
- [x] Demo dự đoán, bảng ví dụ đầu vào/đầu ra và hướng dẫn chạy lại.
- [x] Kiểm thử chia tập, leakage, schema, metrics, artifact, KMeans và stacking.
- [x] Gói bàn giao có source/dữ liệu/model/report và hướng dẫn cài môi trường mới.

Các hướng nghiên cứu tiếp theo (ngoài phạm vi bài hiện tại): nested CV của toàn bộ lựa chọn,
external validation, calibration và ablation từng phép FE. Chưa thực hiện và không được báo cáo là kết quả đã có.

Demo được minh họa bằng bảng ví dụ thật và được kiểm tra qua AppTest.
