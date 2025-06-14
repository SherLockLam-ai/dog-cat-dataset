# Training Dog-Cat Dataset
Mục lục
1. Mô tả file (ml_algo.py)
2. Cấu trúc thư mục
3. Cách sử dụng
4. Thuật toán
5. Kết quả


1. Mô tả file (ml_algo.py)
File ml_algo.py thực hiện các bước sau:

- Tiền xử lý hình ảnh: Đọc hình ảnh từ các thư mục được chỉ định, thay đổi kích thước chúng thành 128x128 pixel và làm phẳng (flatten) chúng thành các vector 1 chiều.
- Chuẩn bị dữ liệu:
  - Tải hình ảnh mèo và chó cho cả tập huấn luyện và kiểm tra.
  - Gán nhãn: 0 cho mèo và 1 cho chó.
  - Ghép các vector hình ảnh và nhãn thành tập dữ liệu huấn luyện và kiểm tra.
- Huấn luyện và đánh giá mô hình: Huấn luyện các thuật toán học máy khác nhau trên tập huấn luyện và đánh giá độ chính xác của chúng trên tập kiểm tra.
- Hàm dự đoán: Bao gồm một hàm predict_image để dự đoán một hình ảnh mèo/chó duy nhất (được huấn luyện với mô hình Hồi quy Logistic).

2. Cấu trúc thư mục
   - trong file
     
3. Cách sử dụng
- Để chạy script, đảm bảo bạn đã cài đặt các thư viện Python cần thiết:
     pip install numpy scikit-learn opencv-python glob2
- Sau đó, chạy script từ terminal:
     python ml_algo.py

4. Thuật toán
Script này huấn luyện và đánh giá các thuật toán học máy sau:

- Perceptron (Perceptron): Một thuật toán phân loại tuyến tính đơn giản.
- K-Nearest Neighbors (KNeighborsClassifier): Một thuật toán phân loại dựa trên khoảng cách.
- Hồi quy Logistic (LogisticRegression): Một mô hình phân loại tuyến tính xác suất.
- Support Vector Machine (SVM) tuyến tính (LinearSVC): Một thuật toán phân loại tuyến tính mạnh mẽ.
- Rừng ngẫu nhiên (RandomForestClassifier): Một thuật toán học tổng hợp (ensemble learning) dựa trên cây quyết định.

5. Kết quả
Script sẽ in độ chính xác của từng mô hình sau khi huấn luyện và dự đoán.
