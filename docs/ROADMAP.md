# Lộ trình thực hiện và giải thích sơ bộ

Luồng chính đã có code chạy. Bạn có thể đọc theo các bước dưới đây để hiểu cách làm,
rồi đối chiếu số đo thật trong reports/baseline_report.md. Phần diễn giải trong notebook và báo cáo đã được hoàn thiện để đối chiếu với code.

## Bạn đang giải quyết điều gì?

Mỗi dòng là một bệnh nhân, với 13 thông tin đầu vào như tuổi, huyết áp, cholesterol
và chỉ số xét nghiệm. Mô hình học từ các bệnh nhân có nhãn để dự đoán lớp 0 hoặc 1.
Đây là **supervised binary classification**: học có giám sát, phân loại hai lớp.

File hiện có gồm 303 dòng, 14 cột không header: 13 feature và 1 target.
Nhãn gốc 0 có 164 mẫu; nhãn lớn hơn 0 có tổng 139 mẫu. Có 4 giá trị thiếu ở `ca`
và 2 ở `thal`. Đây là kết quả kiểm tra file local lúc tạo scaffold, không phải
kết quả huấn luyện; chạy lại `check-data` để xác nhận khi đổi dữ liệu.

## 1. Hiểu dữ liệu trước khi học mô hình

Đọc `DATA_DICTIONARY.md`, mở notebook `00_eda.ipynb`. Kiểm tra kích thước, kiểu dữ
liệu, ô thiếu, dòng trùng, phân bố nhãn và phân bố từng feature. Các số như cp=4
là mã nhóm, không có nghĩa cường độ đau gấp bốn cp=1.

Đầu ra cần có: vài biểu đồ dễ đọc và ghi chú dữ liệu có vấn đề gì. Những quyết
định dùng quan hệ với nhãn để chọn feature chỉ được học trên train.

## 2. Tách dữ liệu thành ba vai trò

- Train: nơi mô hình và các bước tiền xử lý học tham số.
- Validation: đối chiếu và hỗ trợ lựa chọn cách làm.
- Test: đánh giá cuối sau khi đã khóa lựa chọn.

Theo tỉ lệ 80:10:10 và hai lần chia có phân tầng, file này sẽ có 242/30/31 mẫu.
`stratify` giúp tỉ lệ hai lớp gần nhau giữa các tập; seed=42 giúp lặp lại phép chia.
Giữ index gốc để kiểm tra các tập không giao nhau và cùng phép chia được dùng lại.

Ví dụ rò rỉ dữ liệu: nếu lấy median trên toàn 303 người rồi mới chia tập, bạn đã
để thông tin của test tác động vào bước xử lý train. Vì vậy chia trước, fit sau.

## 3. Xây bốn cách biểu diễn cùng dữ liệu

| Biến thể | Cách tạo | Số feature dự kiến |
|---|---|---:|
| Original | Điền thiếu; scale numeric và categorical theo tài liệu | 13 |
| FE | Thêm tỉ lệ/nhóm tuổi; one-hot; bỏ hằng; chọn theo MI | 13 |
| Original + DT | Chọn feature quan trọng từ Original bằng Decision Tree | 10 |
| FE + DT | Chọn feature từ FE bằng Decision Tree | 10 |

Feature engineering là **tạo biểu diễn mới**: ví dụ cholesterol/tuổi.
Feature selection là **giữ lại một số feature** đã có. MI chấm quan hệ của từng
feature với nhãn; DT importance phụ thuộc vào các split của một cây đã học.
Hai cách có thể xếp hạng khác nhau; không cần ép chúng trùng nhau.

Chỉ dùng `fit` trên train rồi `transform` val/test. Transformer phải lưu ngưỡng,
thứ tự và danh sách cột đã học. Với dữ liệu thiếu hoặc ít cột hợp lệ trong một
fold, cần kiểm tra điều kiện Top-K thay vì giả định luôn đủ.

## 4. Học pipeline với một mô hình trước

Bắt đầu bằng `Original + GaussianNB`, rồi thử Decision Tree. Khi luồng đơn chạy
đúng mới mở rộng các model khác. Pipeline gói tiền xử lý và mô hình thành một
đối tượng: dữ liệu thô đi vào, nhãn/xác suất đi ra.

Bảy mô hình chính:

- Naive Bayes: baseline dựa trên xác suất và giả định độc lập có điều kiện.
- KNN: dựa vào các hàng xóm gần, nên nhạy với thang đo.
- Decision Tree: học các luật chia kiểu “nếu/thì”.
- Random Forest: kết hợp nhiều cây học từ các mẫu ngẫu nhiên để giảm phương sai.
- AdaBoost: lần lượt tăng chú ý vào các mẫu bị phân loại sai.
- Gradient Boosting: thêm cây theo hướng giảm hàm mất mát.
- XGBoost: một triển khai gradient boosting có regularization và tối ưu tính toán.

KMeans và stacking được để ở `extensions/` do khác biệt phạm vi giữa các tài liệu;
xem `SOURCES.md`. Không cần hoàn thành chúng trước luồng chính.

## 5. Tìm siêu tham số và benchmark công bằng

Siêu tham số là lựa chọn bạn đặt trước khi học, chẳng hạn độ sâu cây hoặc số hàng
xóm. Với 5-fold CV, train được chia thành 5 phần; mỗi lượt học trên 4 phần và đánh
giá trên phần còn lại. Dùng trung bình các lượt để so cấu hình.

Phải đặt **toàn bộ pipeline** vào CV. Nếu tính MI hoặc chọn Top-K bằng nhãn toàn
train trước CV, fold dùng để đánh giá đã ảnh hưởng vào việc chọn feature.

Chạy 7 model × 4 biến thể = 28 cấu hình theo cùng một quy trình, cùng split và
metrics. Lưu tham số, CV mean/std, điểm validation. Không đổi seed để kiếm điểm đẹp.

## 6. Đánh giá và giải thích kết quả

Sau khi khóa lựa chọn, báo cáo test cho các cấu hình đã định trước. Accuracy cho
biết phần đúng tổng thể; recall lớp 1 cho biết trong những mẫu thực sự có bệnh,
mô hình nhận ra bao nhiêu. Confusion matrix cho thấy rõ false positive/false negative.

Test có 31 mẫu nên sai khác chỉ một người làm accuracy đổi khoảng 3,2 điểm phần
trăm. Chênh lệch nhỏ chưa đủ kết luận một mô hình vượt trội. Phiên bản sơ bộ chạy repeated CV trên train
để nhìn độ dao động của cấu hình đã chọn, báo cáo trung bình và độ lệch chuẩn;
đây là phân tích mô tả vì hyperparameter đã được chọn trên cùng train. Nếu dùng toàn bộ dữ liệu
cho phân tích bổ sung, đừng tiếp tục xem holdout cũ là test độc lập để chọn lại model.

TreeSHAP giúp giải thích **mô hình đã học**: feature nào tác động nhiều nói chung,
và feature nào đẩy dự đoán của một người tăng/giảm. Đây không phải bằng chứng nhân
quả hoặc kết luận y khoa. Bắt đầu với RF + Original; dùng đúng dữ liệu đã transform
và kiểm tra tổng đóng góp khớp đầu ra cùng thang đo của mô hình.

## 7. Hoàn thiện báo cáo và demo

Lưu cả preprocessing và classifier trong một artifact. Demo chỉ nạp pipeline để
dự đoán, tránh xử lý đầu vào theo cách khác lúc train. Chạy local bằng Streamlit;
scaffold không cần public tunnel hoặc tải lại dữ liệu khi mở app.

Đọc `reports/final_report.md`: mục tiêu, dữ liệu, phương pháp, cấu hình, kết quả tự chạy, phân tích và hạn chế; đối chiếu bảng và biểu đồ trong PDF.

## Cách làm một buổi học hiệu quả

Chọn một bước nhỏ, đọc docstring, chạy code đã có,
kiểm tra shape/tên cột/index, rồi ghi quan sát vào notebook. Tránh viết toàn bộ
28 thí nghiệm ngay từ đầu. Một pipeline đơn đúng sẽ giúp bạn mở rộng nhanh hơn.

