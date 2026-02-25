# 🔥 Smart Fire Guard - Hệ Thống Phát Hiện Cháy Sớm Bằng Camera AI (YOLOv11)

> **Đề tài:** Xây dựng hệ thống phát hiện hỏa hoạn sớm dựa trên Camera và AI (Computer Vision).

> **Nguồn tập dữ liệu:** Xây dựng thông qua Roboflow (Roboflow Universe) để tương thích với cấu trúc của YOLO.

## 📖 Giới thiệu

**Smart Fire Guard** là hệ thống phát hiện cháy sớm sử dụng **Camera AI**, khắc phục nhược điểm của các hệ thống báo cháy truyền thống (phụ thuộc cảm biến khói/nhiệt, phản ứng chậm). Thay vì chỉ nhận diện hình ảnh lửa mờ nhạt như các thuật toán Image Classification cũ, phiên bản mới nhất của hệ thống đã được nâng cấp lên **YOLOv11 Object Detection**. Sự thay đổi này giúp vẽ chính xác khung Bounding Box quanh ngọn lửa, nhờ đó phát hiện những đốm lửa rất nhỏ từ xa ngay khi ngọn lửa vừa xuất hiện.

### 🌟 Tính năng chính
* **AI Vision (YOLOv11):** Nâng cấp lên thuật toán phát hiện vật thể tiên tiến nhất hiện nay, cực nhanh và chính xác. Khả năng phát hiện đa dạng (cả Lửa và Khói).
* **Định vị chính xác:** Hiển thị trực tiếp tọa độ và tỉ lệ chính xác (Confidence Score) của ngọn lửa ngay trên khung hình.
* **Chống báo sai:** Yêu cầu phát hiện lửa liên tiếp nhiều frame mới kích hoạt cảnh báo, kết hợp ngưỡng confidence do người dùng tùy chỉnh.
* **Tối ưu hiệu năng:** Chạy inference cực mượt ngay trên các máy tính thông thường (bản `yolo11n`).
* **Dashboard trực quan:** Giao diện Web hiển thị video stream trực tiếp, trạng thái cảnh báo và nhật ký log báo động.

---

## 📊 Biểu đồ Đánh giá Mô hình (Training Results)

Dưới đây là các biểu đồ minh họa thông số kỹ thuật của mô hình YOLOv11 sau khi đã được huấn luyện với tập dữ liệu chất lượng cao.

**1. Hình ảnh dự đoán thực tế trên tập Validation (Validation Predictions):**
> Mô hình có khả năng khoanh vùng chính xác các vị trí lửa (Fire) và khói (Smoke) phức tạp.
![Validation Predictions](runs/detect/fire_yolov11_detection4/val_batch0_pred.jpg)

**2. Ma trận Nhầm lẫn (Confusion Matrix):**
> Tỉ lệ nhận diện đúng của mô hình đối với nhãn Lửa và nhãn Khói.
![Confusion Matrix](runs/detect/fire_yolov11_detection4/confusion_matrix_normalized.png)

**3. Đồ thị Quá trình Huấn luyện (Training Results):**
> Sự tụt giảm độ lỗi (Loss) và tăng cường độ chính xác theo số lượng Epoch.
![Training Results](runs/detect/fire_yolov11_detection4/results.png)

---

## 🛠️ Kiến trúc Hệ thống

### Công nghệ sử dụng (Software Stack)

| Thành phần | Công nghệ |
| :--- | :--- |
| **Backend** | Python, Flask |
| **AI Model** | YOLOv11 (Ultralytics) - Object Detection |
| **Computer Vision** | OpenCV |
| **Frontend** | HTML/CSS/JS, jQuery |

### Luồng hoạt động

```
Webcam → OpenCV đọc frame → Model YOLOv11 phân tích → Vẽ Bounding Box?
                                                       │
                                           ┌───────────┴───────────┐
                                           ▼                       ▼
                                   Có Lửa (≥5 frame)            Không
                                           │                       │
                                     Báo ALARM              Trạng thái Normal
                                     + Ghi Log
```

---

## 🚀 Cài đặt & Hướng dẫn chạy

### Yêu cầu
* **Python 3.8 – 3.11** 
* **Webcam** (laptop có sẵn hoặc webcam USB)

### Bước 1: Thiết lập môi trường ảo (.venv)
Mở Terminal và chạy lần lượt các lệnh sau:

**1. Tạo môi trường ảo:**
```bash
python -m venv .venv
```

**2. Kích hoạt môi trường (Windows PowerShell):**
```bash
.venv\Scripts\activate
```

**3. Cài đặt các thư viện cần thiết:**
> Lưu ý cài đặt thư viện ultralytics thay cho tensorflow cũ.
```bash
pip install ultralytics Flask opencv-python numpy
```

### Bước 2: Training AI Model (Hoặc dùng file có sẵn)

Hệ thống đã có cấu trúc train tự động nếu bạn tải dataset từ Roboflow về thư mục `dataset_yolo`.
Để bắt đầu train, gọi lệnh:
```bash
python train_yolo.py
```
Sau khi train xong, file trọng số (model weights) tốt nhất sẽ nằm ở thư mục `runs/detect/.../weights/best.pt`. (Xác nhận lại đường dẫn được chỉ định trong file `app.py`).

### Bước 3: Chạy ứng dụng web
```bash
python app.py
```

Truy cập trình duyệt tại địa chỉ: **http://127.0.0.1:5000**

---

## ⚙️ Tùy chỉnh tham số (File app.py)

Bạn có thể chỉnh lại các tham số ở đầu file `app.py`:

```python
CONFIDENCE_THRESHOLD = 0.3    # Độ tin cậy tối thiểu để vẽ khung Lửa (0.3 = 30%)
CONSECUTIVE_FRAMES = 5        # Số frame liên tiếp phải có ô vuông lửa thì mới hú còi báo động
PREDICT_EVERY_N_FRAMES = 3    # Chạy AI mỗi N frame (tăng lên nếu máy yếu/lag)
LOG_COOLDOWN = 5              # Thời gian nghỉ ghi log tránh bị spam chữ (giây)
```

---

## 📂 Cấu trúc Thư mục

```
SmartFireGuard/
│
├── app.py                # Web server Flask + Logic Camera & YOLO
├── train_yolo.py         # Script tự huấn luyện model YOLO mới
├── data.yaml             # File cấu hình đường dẫn Dataset YOLO
├── requirements.txt      # (Nên cập nhật: ultralytics thay thế tensorflow)
│
├── dataset_yolo/         # Nơi để Dataset tải về từ phần mềm bên thứ 3
│   ├── images/
│   └── labels/
│
├── runs/                 # Chứa các file kết quả và trọng số best.pt sinh ra khi train
│
├── templates/
│   └── index.html        # Giao diện Dashboard Web
│
└── README.md
```

---

## ⚠️ Khắc phục lỗi thường gặp

**1. Model báo sai liên tục (báo người/xe cộ thành lửa):**
> Điều này là MÔ HÌNH CHƯA TRAIN. Hiện tại model đang load `yolo11n.pt` gốc của YOLO chỉ dùng để nhận diện 80 các lớp người và động vật. Bạn bắt buộc phải cho chạy `python train_yolo.py` và đổi đường dẫn trong `app.py` để trỏ tới `best.pt`.

**2. Lỗi `FileNotFoundError` khi load YOLO:**
> Sửa lại đường dẫn nạp model ở đầu file `app.py` trỏ đúng vào thư mục `runs/detect/...` mới nhất của bạn.

**3. Báo cháy chập chờn:**
> Hạ thấp tham số `CONFIDENCE_THRESHOLD` trong `app.py` xuống `0.2`.

---

## 👨‍💻 Tác giả
Họ và tên: Lê Phước Hậu

Lớp/MSSV: 2033221314 - Nhóm 16

Dự án: Đồ án IoT/AI - Smart Fire Guard (Bản nâng cấp YOLOv11)