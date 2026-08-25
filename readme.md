# Flask Video & Camera  With YOLO

เว็บแอปพลิเคชันสำหรับประมวลผลและสตรีมวิดีโอแบบ Real-time ด้วยโมเดล YOLOv8 บน Flask Framework

---

## 📁 Project Structure

โครงสร้างไฟล์และโฟลเดอร์ภายในโปรเจกต์:

```text
├── app.py
├── controllers
│   ├── __pycache__
│   │   ├── main_controller.cpython-313.pyc
│   │   ├── upload_controller.cpython-313.pyc
│   │   └── webcam_controller.cpython-313.pyc
│   ├── upload_controller.py
│   └── webcam_controller.py
├── readme.md
├── requirements.txt
├── static
│   └── uploads
│       ├── 13002160_1920_1080_60fps.mp4
│       ├── 18437773-uhd_3840_2160_50fps.mp4
│       ├── 795.mov
│       └── car.mp4
├── templates
│   ├── camera.html
│   ├── index.html
│   ├── layout.html
│   └── upload.html
├── utils
│   ├── __pycache__
│   │   └── monad.cpython-313.pyc
│   └── monad.py
└── yolov8n.pt