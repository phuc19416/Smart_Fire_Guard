import os
from ultralytics import YOLO

def main():
    print("=== BẮT ĐẦU HUẤN LUYỆN YOLOv11 ===")
    
    # Khởi tạo mô hình bản n (nano - nhẹ và nhanh nhất)
    # Nếu lần đầu chạy, Ultralytics sẽ tự động tải file yolo11n.pt về máy
    model = YOLO('yolo11n.pt')
    
    # Bắt đầu huấn luyện
    # Cấu hình data.yaml chứa đường dẫn tới dataset của bạn
    # imgsz=640 là kích thước ảnh chuẩn của YOLO
    results = model.train(
        data='dataset_yolo/data.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        name='fire_yolov11_detection'
    )
    
    print("=== HUẤN LUYỆN HOÀN TẤT ===")
    print("Mô hình tốt nhất được lưu tại: runs/detect/fire_yolov11_detection/weights/best.pt")

if __name__ == '__main__':
    main()
