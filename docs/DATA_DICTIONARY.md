# Cleveland: từ điển dữ liệu

CSV gốc không header, cột đúng thứ tự bảng dưới. `?` hoặc ô trống biểu diễn thiếu.
Tên `target` trong code tương ứng `num` trong mô tả bộ dữ liệu gốc.

| Cột | Ý nghĩa theo tài liệu môn học | Kiểu xử lý |
|---|---|---|
| age | Tuổi, năm | numeric |
| sex | 0=nữ, 1=nam | categorical |
| cp | 1=đau thắt ngực điển hình; 2=không điển hình; 3=đau không do tim; 4=không triệu chứng | categorical |
| trestbps | Huyết áp tâm thu lúc nghỉ, mmHg | numeric |
| chol | Cholesterol huyết thanh, mg/dL | numeric |
| fbs | Đường huyết lúc đói >120 mg/dL: 0=sai, 1=đúng | categorical |
| restecg | 0=bình thường; 1=bất thường ST-T; 2=phì đại thất trái | categorical |
| thalach | Nhịp tim tối đa đạt được, lần/phút | numeric |
| exang | Đau ngực khi gắng sức: 0=không, 1=có | categorical |
| oldpeak | Mức trầm ST do gắng sức so với lúc nghỉ | numeric |
| slope | 1=dốc lên; 2=bằng phẳng; 3=dốc xuống | categorical |
| ca | Số mạch máu chính quan sát, 0–3 | categorical |
| thal | Kết quả thallium stress test: 3=bình thường; 6=khuyết tật cố định; 7=có thể hồi phục | categorical |
| target | Gốc: 0–4; trong project: 0 nếu gốc=0, 1 nếu gốc>0 | label |

Năm numeric và tám categorical là quy ước của tài liệu project. Categorical vẫn
được lưu bằng mã số trong CSV. Không tự đổi mã theo một bản Cleveland khác trên mạng.

Đặc trưng sinh mới: `chol_per_age=chol/age`, `bps_per_age=trestbps/age`,
`hr_ratio=thalach/age`, `age_bin` theo quantile của tuổi **training subset**.
Đây là các biểu diễn thực nghiệm; không tự gán ý nghĩa chỉ số y khoa chuẩn cho chúng.

