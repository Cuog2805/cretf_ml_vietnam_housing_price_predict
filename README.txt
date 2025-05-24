Service dự đoán giá bất động sản Việt Nam:
- Xử lý dữ liệu: Làm sạch và chuẩn hóa địa chỉ Việt Nam
- Model học máy: RandomForestRegression
- API REST: FastAPI service

Công nghệ sử dụng:
- Machine Learning: scikit-learn, pandas, numpy
- API: FastAPI, uvicorn
- Lưu trữ model: joblib serialization

Input Features (Đặc trung đầu vào):

Trường          | Kiểu    | Mô tả                            | Ví dụ          | Bắt buộc
----------------|---------|----------------------------------|----------------|--------
District        | string  | Tên quận/huyện (tự động làm sạch)| "Lý Nhân"      | Có
Province        | string  | Tên tỉnh/thành (tự động làm sạch)| "Hà Nam"       | Có
Area            | number  | Diện tích (m²)                   | 84             | Có
Frontage        | number  | Mặt tiền (m)                     | 5              | Không
Access Road     | number  | Đường vào (m)                    | 8              | Không
Direction       | string  | Hướng nhà                        | "Đông - Bắc"   | Không
Type            | string  | Loại BĐS                         | "Nhà"/"Căn hộ" | Có
Floors          | number  | Số tầng                          | 4              | Không
Bedrooms        | number  | Số phòng ngủ                     | 3              | Không
Bathrooms       | number  | Số phòng tắm                     | 2              | Không

Làm sạch địa chỉ tự động:

Quận/Huyện: Bỏ tiền tố Huyện/ Thành phố/ Quận/ Thị xã
"Huyện Lý Nhân"    → "Lý Nhân"

Tỉnh/Thành phố: Bỏ tiền tố Tỉnh/ Thành phố
"Tỉnh Hà Nam"      → "Hà Nam"

TRAINING MODEL:
1. Tải dữ liệu từ file JSON
2. Phân tích địa chỉ: Tách quận/huyện và tỉnh/thành từ địa chỉ
3. Chia Tier theo khu vực
3. Xác định loại BĐS: Từ hướng nhà/ban công
4. Làm sạch dữ liệu: Loại bỏ records không hợp lệ và outliers
5. Feature engineering: Tạo categorical encodings
6. Training model: RandomForestRegression với parameters tối ưu
7. Validation: Đánh giá bằng R2 score và error metrics
8. Lưu model: Persist model và encoders

API SERVICE
Service: http://localhost:8083
Endpoints:
1. DỰ ĐOÁN GIÁ - POST /predict
2. Swagger UI: http://localhost:8083/docs
