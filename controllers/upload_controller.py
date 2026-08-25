import os
from flask import Blueprint, render_template, request, Response
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO

from utils.monad import Result

# ตั้งชื่อ Blueprint (หรือจะมองว่าเป็น Upload Controller)
upload_controller = Blueprint("upload_controller", __name__)
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

model = YOLO("yolov8n.pt")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Monad Validation Functions
def check_file_exists(request_files):
    if "video_file" not in request_files:
        return Result.failure("ไม่พบข้อมูลไฟล์ใน Request")
    file = request_files["video_file"]
    if file.filename == "":
        return Result.failure("ไม่ได้เลือกไฟล์วิดีโอ")
    return Result.success(file)


def check_file_extension(file):
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return Result.failure("รองรับเฉพาะไฟล์วิดีโอ (mp4, avi, mov, mkv) เท่านั้น")
    return Result.success((file, filename))


def save_file_to_disk(data):
    file, filename = data
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return Result.success(filename)


# utils
@upload_controller.route("/video_feed/<filename>")
def video_feed(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    # ประกาศ Generator ซ้อนไว้ข้างใน Route นี้ตัวเดียวจบ
    def generate():
        cap = cv2.VideoCapture(filepath)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # รัน YOLO และวาดกรอบ
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()

            # แปลงภาพเป็น JPEG
            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue

            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
        cap.release()

    # สั่ง Return Response โดยเรียกใช้ Generator ด้านในทันที
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# Route Definitions
@upload_controller.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html", title="สตรีมวิดีโอ Real-time - YOLO")


@upload_controller.route("/upload-video", methods=["POST"])
def upload_video():
    result = (
        Result.success(request.files)
        .bind(check_file_exists)
        .bind(check_file_extension)
        .bind(save_file_to_disk)
    )

    if result.is_success:
        filename = result.value
        return render_template("upload.html", title="อัปโหลดวิดีโอ", filename=filename)
    else:
        return render_template("upload.html", title="อัปโหลดวิดีโอ", error=result.error)
