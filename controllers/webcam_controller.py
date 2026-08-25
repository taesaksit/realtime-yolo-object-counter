from flask import Blueprint, render_template, Response
import cv2
from ultralytics import YOLO

# Webcam Controller
webcam_controller = Blueprint('webcam_controller', __name__)
model = YOLO("yolov8n.pt")

@webcam_controller.route("/camera", methods=["GET"])
def camera_page():
    return render_template("camera.html", title="เปิดกล้องสด Real-time - YOLO")

def generate_webcam_stream():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
    cap.release()

@webcam_controller.route("/webcam_feed")
def webcam_feed():
    return Response(
        generate_webcam_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )