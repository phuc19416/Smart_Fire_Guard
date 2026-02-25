import cv2
import time
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- BIẾN TOÀN CỤC ---
system_status = {
    "fire_detected": False, "alarm": False, "log": []
}

# --- 1. AI MODEL LOAD ---
try:
    # Mặc định load model YOLOv11 nano (tự động tải nếu chưa có)
    # Nếu bạn đã tự train mô hình, hãy đổi thành: YOLO('runs/detect/fire_yolov11_detection/weights/best.pt')
    model = YOLO(r'runs\detect\fire_yolov11_detection5\weights\best.pt') 
    YOLO_MODE = True
    print("AI Model (YOLO) Loaded!")
except Exception as e:
    print(f"Lỗi load model YOLO: {e}\nDùng chế độ giả lập AI (Color detection)!")
    model = None
    YOLO_MODE = False

# --- 2. CAMERA & AI PROCESSING ---
CONFIDENCE_THRESHOLD = 0.3    # Độ tin cậy tối thiểu để nhận diện là lửa (0.3 = 30%)
CONSECUTIVE_FRAMES = 5        # Phải phát hiện lửa liên tiếp N frame mới báo alarm
PREDICT_EVERY_N_FRAMES = 3    # Chỉ chạy AI mỗi N frame để giảm lag
LOG_COOLDOWN = 5              # Thời gian nghỉ ghi log tránh spam (giây)

fire_streak = 0               
last_log_time = 0

def generate_frames():
    global fire_streak, last_log_time
    camera = cv2.VideoCapture(0)
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success: break
        
        frame_count += 1
        label_text = "Normal"
        color = (0, 255, 0)
        max_confidence = 0.0
        fire_detected_this_frame = False

        if frame_count % PREDICT_EVERY_N_FRAMES == 0:
            if YOLO_MODE:
                # Chạy YOLO Inference
                results = model(frame, verbose=False)[0]
                
                # Biến cờ kiểm tra xem trong danh sách các box có box nào là lửa không
                # LƯU Ý: Với yolo11n.pt gốc (COCO dataset), class 'fire' không có sẵn. 
                # Model pretrained COCO chỉ nhận diện người, chó, xe... vv
                # Đoạn code này được giả định sẽ chạy với model BẠN ĐÃ TỰ TRAIN trên file dataset lửa
                # Trong bộ dataset tự train, class lửa thường có id = 0.
                
                for box in results.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Giả sử class lửa của chúng ta có index = 0 sau khi train
                    # Nếu đang chạy yolo11n.pt mặc định, nó sẽ nhận 'person' là 0 (sẽ báo cháy nhầm người)
                    if cls == 0 and conf >= CONFIDENCE_THRESHOLD:
                        fire_detected_this_frame = True
                        if conf > max_confidence:
                            max_confidence = conf
                        
                        # Vẽ Bounding Box đỏ quanh mục tiêu
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"FIRE {conf*100:.0f}%", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Fallback: Dùng color threshold đơn giản
                import numpy as np
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower = np.array([0, 50, 50])
                upper = np.array([35, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                if cv2.countNonZero(mask) > 5000:
                    fire_detected_this_frame = True
                    max_confidence = 0.99
            
            # Cập nhật số counter liên tiếp
            if fire_detected_this_frame:
                fire_streak += 1
            else:
                fire_streak = 0

            # Báo Alarm nếu đủ số frame
            if fire_streak >= CONSECUTIVE_FRAMES:
                label_text = "FIRE DETECTED"
                color = (0, 0, 255)
                system_status['fire_detected'] = True
                system_status['alarm'] = True

                if time.time() - last_log_time > LOG_COOLDOWN:
                    system_status['log'].append(
                        f"{time.strftime('%H:%M:%S')} - ALARM: Phát hiện LỬA! ({max_confidence*100:.0f}%)")
                    last_log_time = time.time()
            else:
                system_status['fire_detected'] = False
                system_status['alarm'] = False

        # Chỉ vẽ chữ cảnh báo tổng quát ở góc nếu frame không chia hết cho N hoặc có báo động
        if system_status['alarm']:
            cv2.putText(frame, "ALARM: FIRE DETECTED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "STATUS: NORMAL", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 3. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def get_data():
    return jsonify(system_status)

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)