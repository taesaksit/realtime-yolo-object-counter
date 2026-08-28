from flask import Blueprint, render_template, Response, request
import cv2
from ultralytics import YOLO

# Webcam Controller
webcam_controller = Blueprint("webcam_controller", __name__)

model = YOLO("yolov8n.pt")


def get_available_cameras(max_cameras=10):
    """
    ตรวจสอบว่ามีกล้องตัวไหนเปิดใช้งานได้บ้าง
    """

    cameras = []

    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)

        if cap.isOpened():
            cameras.append({
                "index": index,
                "name": f"Camera {index}"
            })

        cap.release()

    return cameras


@webcam_controller.route("/camera", methods=["GET"])
def camera_page():
    cameras = get_available_cameras()

    return render_template(
        "camera.html",
        title="เปิดกล้องสด Real-time - YOLO",
        cameras=cameras
    )


def generate_webcam_stream(camera_index):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        return

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # YOLO Detection
            results = model(frame, verbose=False)

            annotated_frame = results[0].plot()

            # Encode JPEG
            ret, buffer = cv2.imencode(".jpg", annotated_frame)

            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    finally:
        cap.release()


@webcam_controller.route("/webcam_feed")
def webcam_feed():
    camera_index = request.args.get("camera", default=0, type=int)

    return Response(
        generate_webcam_stream(camera_index),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )