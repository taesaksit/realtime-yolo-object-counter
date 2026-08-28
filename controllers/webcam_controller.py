import os
import time

import cv2
import torch
from flask import Blueprint, Response, render_template, request
from ultralytics import YOLO, solutions
from utils.device import get_device
from services.model_service import get_available_models, load_model


webcam_controller = Blueprint("webcam_controller", __name__)

MODEL_FOLDER = "./models"
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360
DEVICE = get_device()


def get_available_cameras():

    cameras = []

    for index in range(10):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            cameras.append(index)

        cap.release()

    return cameras


@webcam_controller.route("/webcam", methods=["GET"])
def webcam_page():

    models_list = get_available_models()

    return render_template(
        "webcam.html",
        title="Webcam YOLO Object Counter",
        models=models_list,
    )


@webcam_controller.route("/cameras", methods=["GET"])
def cameras():

    return {"cameras": get_available_cameras()}


@webcam_controller.route("/models", methods=["GET"])
def models():

    return {"models": get_available_models()}


def get_line_points():

    try:
        x1 = int(request.args.get("x1", 100))

        y1 = int(request.args.get("y1", 180))

        x2 = int(request.args.get("x2", 500))

        y2 = int(request.args.get("y2", 180))

    except (TypeError, ValueError):
        x1 = 100
        y1 = 180

        x2 = 500
        y2 = 180

    x1 = max(0, min(x1, PROCESS_WIDTH - 1))
    y1 = max(0, min(y1, PROCESS_HEIGHT - 1))
    x2 = max(0, min(x2, PROCESS_WIDTH - 1))
    y2 = max(0, min(y2, PROCESS_HEIGHT - 1))

    return [(x1, y1), (x2, y2)]


@webcam_controller.route("/webcam_feed", methods=["GET"])
def webcam_feed():

    try:
        camera_index = int(request.args.get("camera", 0))

    except (TypeError, ValueError):
        camera_index = 0

    mode = request.args.get("mode", "preview")
    model_name = request.args.get("model", "")
    line_point = get_line_points()

    model = None

    if mode == "detection":
        if not model_name:
            return Response("Model is required", status=400)

        model = load_model(model_name)

        if model is None:
            return Response("Model not found", status=404)

        print(f"[INFO] Model: {model_name}")
        print(f"[INFO] Device: {DEVICE}")
        print(f"[INFO] Region: {line_point}")

    def render():

        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open webcam {camera_index}")

            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        counter = None

        if mode == "detection":
            counter = solutions.ObjectCounter(
                show=False,
                model=model,
                region=line_point,
                device=DEVICE,
                show_in=True,
                show_out=True,
                classes=None,
                conf=0.25,
                iou=0.45,
                show_conf=True,
                show_labels=True,
                show_boxes=True,
                line_width=2,
                tracker="botsort.yaml",
                imgsz=416,
                verbose=False,
            )

        prev_time = time.perf_counter()
        fps = 0.0

        try:
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    print("[ERROR] Failed to read webcam frame.")

                    break

                frame = cv2.resize(
                    frame, (PROCESS_WIDTH, PROCESS_HEIGHT), interpolation=cv2.INTER_AREA
                )

                if mode == "preview":
                    annotated_frame = frame

                else:
                    results = counter(frame)

                    annotated_frame = results.plot_im

                current_time = time.perf_counter()

                elapsed = current_time - prev_time

                if elapsed > 0:
                    fps = 1.0 / elapsed

                prev_time = current_time

                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if mode == "detection" and counter is not None:
                    cv2.putText(
                        annotated_frame,
                        f"IN: {results.in_count}",
                        (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.putText(
                        annotated_frame,
                        f"OUT: {results.out_count}",
                        (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    total_count = results.in_count + results.out_count

                    cv2.putText(
                        annotated_frame,
                        f"TOTAL: {total_count}",
                        (15, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                ret, buffer = cv2.imencode(
                    ".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )

                if not ret:
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

        except GeneratorExit:
            print("[INFO] Webcam stream stopped.")

        except Exception as e:
            print(f"[ERROR] Webcam stream: {e}")

        finally:
            cap.release()

            print(f"[INFO] Webcam {camera_index} released.")

    return Response(render(), mimetype=("multipart/x-mixed-replace; boundary=frame"))
