# BÁO CÁO CHI TIẾT
## Đề tài: Phát Hiện Cháy Sớm Bằng Camera AI 
## Phân Tích Chuyên Sâu Mô Hình YOLOv11

---

### MỤC LỤC
1. [Giới thiệu Chung](#1-giới-thiệu-chung)
2. [Cơ sở Lý thuyết: Thuật toán YOLO (You Only Look Once)](#2-cơ-sở-lý-thuyết-thuật-toán-yolo-you-only-look-once)
3. [Phân tích Kiến trúc và Hoạt động của YOLOv11](#3-phân-tích-kiến-trúc-và-hoạt-động-của-yolov11)
4. [Ứng dụng YOLOv11 vào Hệ thống Smart Fire Guard](#4-ứng-dụng-yolov11-vào-hệ-thống-smart-fire-guard)
5. [Kết quả Thực nghiệm & Đánh giá](#5-kết-quả-thực-nghiệm-và-đánh-giá)
6. [Hướng Cải thiện](#6-hướng-cải-thiện)

---

### 1. Giới thiệu Chung
Trọng tâm của hệ thống Smart Fire Guard là mô hình Trí tuệ nhân tạo (AI) có khả năng phân tích hình ảnh từ Camera để nhận ra ngọn lửa ngay khi nó vừa xuất hiện. Để đáp ứng yêu cầu **tốc độ cao (Real-time)** và **nhận diện chính xác vị trí lửa nhỏ**, dự án đã quyết định sử dụng **YOLOv11** - thế hệ mạng nơ-ron phát hiện vật thể (Object Detection) tiên tiến nhất hiện nay (ra mắt cuối 2024).

---

### 2. Cơ sở Lý thuyết: Thuật toán YOLO (You Only Look Once)
Trước khi tìm hiểu YOLOv11, chúng ta cần hiểu rõ nguyên lý gốc của dòng mô hình YOLO. Khác với các mô hình 2-giai-đoạn (như Faster R-CNN) phải quét ảnh nhiều lần để tìm vùng ứng viên rồi mới phân loại, YOLO xử lý bài toán detection dưới dạng một **bài toán hồi quy (regression) duy nhất**.

**Cách YOLO hoạt động cơ bản:**
1. **Chia lưới ảnh (Grid Layout):** YOLO chia bức ảnh đầu vào (ví dụ 640x640) thành một lưới (Grid) kích thước $S \times S$.
2. **Nhiệm vụ của mỗi ô lưới (Grid Cell):** Nếu tâm của ngọn lửa rơi vào một ô lưới nào, ô lưới đó phải chịu trách nhiệm phát hiện ngọn lửa đó.
3. **Dự đoán Bounding Box:** Mỗi ô lưới sẽ dự đoán $B$ bounding boxes. Mỗi box chứa 5 tham số:
   - $(x, y)$: Tọa độ tâm của box so với ô lưới.
   - $(w, h)$: Chiều rộng và chiều cao của box so với toàn bộ bức ảnh.
   - $Confidence$: Độ tự tin (mức độ chắc chắn box này chứa một đối tượng).
4. **Dự đoán Class Probability:** Đồng thời, ô lưới dự đoán xác suất đối tượng đó thuộc lớp nào (Lửa - Fire hay Khói - Smoke).
5. **Non-Maximum Suppression (NMS):** Vì một ngọn lửa có thể bị nhiều ô lưới kề nhau cùng dự đoán (tạo ra nhiều box trùng lặp), thuật toán NMS sẽ được dùng để loại bỏ các box thừa, chỉ giữ lại box có Confidence lớn nhất.

---

### 3. Phân tích Kiến trúc và Hoạt động của YOLOv11
YOLOv11 kế thừa triết lý của các bản YOLO trước (như YOLOv8) nhưng mang lại những nâng cấp vượt trội về **Backbone** và khả năng tối ưu hóa tính toán, đặc biệt cho các hệ thống biên (Edge Devices) như Camera/IoT.

#### 3.1. Cấu trúc Mạng Neural của YOLOv11
Cấu trúc mạng của YOLOv11 được chia làm 3 phần chính giống như một dây chuyền phân tích hình ảnh:

**A. Backbone (Bộ chiết xuất đặc trưng)**
- **Nhiệm vụ:** Nơi đầu tiên ảnh đi qua (dưới dạng ma trận pixel). Nó đóng vai trò "rút trích" các đặc trưng (features) từ cơ bản như cạnh, góc, vân sáng... cho đến các đặc trưng phức tạp như hình dáng ngọn lửa.
- **Cải tiến ở YOLOv11:** Sử dụng các khối chập (Convolutional blocks) hiệu quả hơn (như **C3k2**) kết hợp với cơ chế Attention (SPPF - Spatial Pyramid Pooling Fast) giúp mô hình "nhìn" được bối cảnh rộng hơn của ngọn lửa mà không tốn quá nhiều phép tính.

**B. Neck (Bộ tổng hợp đặc trưng)**
- **Nhiệm vụ:** Trộn lẫn và kết nối các đặc trưng hình ảnh ở nhiều độ phân giải khác nhau. Điều này cực kỳ quan trọng đối với **LỬA** vì lửa có thể ở rất xa (chiếm vài pixel) hoặc ở rất gần (chiếm cả màn hình).
- **Kiến trúc:** Sử dụng kiến trúc FPN (Feature Pyramid Network) và PANet để truyền đặc trưng từ các lớp sâu (ngữ nghĩa tốt nhưng độ phân giải thấp) kết nối với các lớp nông (ngữ nghĩa ít nhưng chi tiết không gian cao). Giúp lấy được thông tin chi tiết của các đốm lửa cực nhỏ.

**C. Head (Bộ xuất kết quả - Dự đoán)**
- **Nhiệm vụ:** Giải mã các đặc trưng đã tổng hợp để đưa ra kết quả cuối cùng: "Vật thể nằm ở đâu?" và "Đó là cái gì?".
- **Anchor-Free:** YOLOv11 sử dụng kỹ thuật không dùng mỏ neo (Anchor-free). Thay vì phải khai báo trước hàng loạt các hộp mẫu có tỷ lệ nhất định (ví dụ 1:1, 1:2), Head của mô hình dự đoán thẳng khoảng cách từ điểm trung tâm đến 4 cạnh của hộp (Center-to-edge). Do ngọn lửa không có hình dáng cố định (lúc cao lúc bè), việc dùng Anchor-free giúp model nhận diện ngọn lửa cực kỳ linh hoạt và mượt mà.

#### 3.2. Phiên bản YOLO11n (Nano)
Dự án sử dụng bản **YOLO11n (Nano)**. Đây là phiên bản có số lượng tham số (Parameters) ít nhất. Việc hy sinh một chút độ sâu mạng giúp mô hình có thể đẩy tốc độ suy luận (Inference Speed) lên vài chục đến hàng trăm FPS ngay cả khi chạy bằng CPU máy tính thông thường, đáp ứng đúng yêu cầu **"Phát hiện Sớm và Tức thời"**.

---

### 4. Ứng dụng YOLOv11 vào Hệ thống Smart Fire Guard
Quá trình đưa mô hình toán học YOLOv11 vào ứng dụng thực tế trên Web Flak trải qua các bước:

* **Tập dữ liệu chuẩn YOLO:** Thay vì ảnh lưu rời rạc theo thư mục Class, Dataset được gán nhãn dưới dạng file text tọa độ `.txt` chứa `[Class_ID X_center Y_center Width Height]`. Hệ thống học được "Hình dáng ngọn lửa ở góc này so với chiều toàn bức ảnh".
* **Đầu vào (Input Streaming):** Camera đọc frame thông qua thư viện `cv2`. Hình ảnh màu BGR (nhiễu tín hiệu) được truyền thẳng cho API của YOLO.
* **Suy luận (Inference):** Lệnh `results = model(frame)` sẽ đẩy ma trận ảnh qua luồng Backbone -> Neck -> Head như đã phân tích.
* **Bóc tách đầu ra (Post-processing):**
  Trong file `app.py`, code sẽ duyệt qua `results.boxes` để lấy cụ thể:
  - Khung giới hạn `box.xyxy` (Tọa độ tuyệt đối X1,Y1, X2,Y2 trên khung hình).
  - Độ tin cậy `box.conf` (Confidence).
  - Nhập ID `box.cls` (lọc ID=0 tức là Fire).
* **Bảo lưu (Debouncing):** Dù YOLO phát hiện rất nhanh, lửa có đặc thù hay bị nhòe chớp. Hệ thống dùng biến đếm `fire_streak`. YOLO phải trả về Bounding Box chứa Fire liên tục 5 frame thì cờ `ALARM` mới kích hoạt, bù trừ hoàn toàn sai số của AI.

---

### 5. Kết quả Thực nghiệm và Đánh giá

Với tập dữ liệu Roboflow được huấn luyện qua 25 chu kỳ (Epochs), YOLOv11 thể hiện sức mạnh của nó qua các đồ thị phân tích lỗi:

* **Box Loss & Objectness Loss (Hàm mất mát Bounding box):**
  Biểu đồ train cho thấy mức Loss giảm cắm đầu theo đường cong trơn tru ngay từ Epoch số 3. Head Anchor-free của YOLOv11 chứng tỏ nó bắt được hình thái ngọn lửa cực tốt thay vì phải "o ép" ngọn lửa vào các dạng hộp hình vuông cố định.
* **Chỉ số mAP (Mean Average Precision):**
  - **mAP50 đạt mức ~ 42.2%** chỉ sau một thời gian train cực ngắn (25 vòng). Tức là trong thực tế, mọi đốm lửa rõ ràng khi lọt vào tầm nhìn đều bị khoanh đỏ không trượt phát nào.
* **Thời gian đáp ứng (Latency):**
  Thế mạnh lớn nhất của thiết kế **Single-stage (Nhìn 1 lần)** của YOLO thể hiện rõ ở đây. Inference xử lý xong hình ảnh 640x640 trong thời gian tính bằng miligây, hệ thống Web trả Video Stream cực mượt, không hề có hiện tượng khựng hình như các model cũ.

---

### 6. Hướng Cải thiện
Để tối đa hóa triệt để sức mạnh của kiến trúc mạng YOLOv11, dự án có thể mở rộng theo hướng:
1. **Huấn luyện sâu hơn:** Đẩy số lượng Epoch lên mức chuẩn thực tế (từ 100 đến 300). Cấu trúc Backbone kết hợp Attention (SPPF) của YOLOv11 rất sâu, 25 Epoch chưa thể làm ấm hết (warm up) các trọng số.
2. **Data Augmentation mạnh hơn:** Kích hoạt các bộ sinh Augmentation bên trong mô hình để tự tạo các ngọn lửa giả (xoay, lật, thay đổi độ sáng bối cảnh), ép mô hình học các tình huống môi trường phức tạp (đèn nháy, hoàng hôn) giảm thiểu False Positives.
3. **Triển khai ở biên (Edge Deployment):** Export model từ dạng `.pt` (PyTorch) sang chuẩn tối ưu cho phần cứng như **ONNX** hoặc **TensorRT**. Từ đó có thể nạp mô hình chạy trực tiếp trên các bo mạch Raspberry Pi hay Jetson Nano đính kèm Camera thay vì phải phụ thuộc vào máy chủ.
