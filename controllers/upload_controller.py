import os
import time

import cv2
import torch
from flask import Blueprint, Response, render_template, request
from ultralytics import YOLO, solutions
from werkzeug.utils import secure_filename

from utils.monad import Result
from utils.device import get_device
from services.model_service import get_available_models, load_model, get_default_model

upload_controller = Blueprint("upload_controller", __name__)


UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
DEVICE = get_device()
default_model = get_default_model()


# validate
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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


# env validate


@upload_controller.route("/video_feed/<filename>")
def video_feed(filename):

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    try:
        x1 = int(request.args.get("x1", 100))
        y1 = int(request.args.get("y1", 100))

        x2 = int(request.args.get("x2", 500))
        y2 = int(request.args.get("y2", 100))

        line_point = [
            (x1, y1),
            (x2, y2),
        ]

    except (TypeError, ValueError):
        line_point = [
            (100, 100),
            (500, 100),
        ]

    requested_model = request.args.get("model", default_model=default_model)

    available_models = get_available_models()

    if requested_model not in available_models:
        if default_model in available_models:
            requested_model = default_model

        elif available_models:
            requested_model = available_models[0]

        else:
            return "ไม่พบ YOLO model ในโฟลเดอร์ models", 404

    def generate():

        cap = cv2.VideoCapture(filepath)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {filepath}")

            return

        selected_model = load_model(requested_model)

        if selected_model is None:
            print(f"[ERROR] Cannot load model: {requested_model}")

            cap.release()

            return

        print(
            f"[INFO] Starting detection"
            f" | Model: {requested_model}"
            f" | Device: {DEVICE}"
            f" | Line: {line_point}"
        )

        counter = solutions.ObjectCounter(
            show=False,
            model=selected_model,
            region=line_point,
            device=DEVICE,
            imgsz=640,
            show_in=True,
            show_out=True,
            classes=None,
            conf=0.25,
            iou=0.45,
            show_conf=True,
            show_labels=True,
            line_width=2,
            tracker="botsort.yaml",
            verbose=False,
        )

        prev_time = time.perf_counter()

        try:
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    break

                frame = cv2.resize(
                    frame,
                    (
                        PROCESS_WIDTH,
                        PROCESS_HEIGHT,
                    ),
                )

                results = counter(frame)

                current_time = time.perf_counter()

                elapsed = current_time - prev_time

                fps = 1 / elapsed if elapsed > 0 else 0

                prev_time = current_time

                annotated_frame = results.plot_im

                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    annotated_frame,
                    f"IN: {results.in_count}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    annotated_frame,
                    f"OUT: {results.out_count}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    annotated_frame,
                    f"Model: {requested_model}",
                    (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                ret, buffer = cv2.imencode(
                    ".jpg",
                    annotated_frame,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        80,
                    ],
                )

                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

        except GeneratorExit:
            print("[INFO] Video stream stopped.")

        finally:
            cap.release()

            print("[INFO] VideoCapture released.")

    return Response(
        generate(),
        mimetype=("multipart/x-mixed-replace; boundary=frame"),
    )


@upload_controller.route("/upload", methods=["GET"])
def upload_page():

    models = get_available_models()

    return render_template(
        "upload.html",
        title="สตรีมวิดีโอ Real-time - YOLO",
        models=models,
        default_model=default_model,
    )


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

        models = get_available_models()

        return render_template(
            "upload.html",
            title="อัปโหลดวิดีโอ",
            filename=filename,
            models=models,
            default_model=default_model,
        )

    else:
        models = get_available_models()

        return render_template(
            "upload.html",
            title="อัปโหลดวิดีโอ",
            error=result.error,
            models=models,
            default_model=default_model,
        )
