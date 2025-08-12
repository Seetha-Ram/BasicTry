import time
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model from local file
model = YOLO("yolov8n.pt")
MIN_CONF = 0.25

def decide_direction_from_results(results, frame_width):
    """Decide movement direction based on object positions."""
    zone = {"left": False, "center": False, "right": False}
    if results is None or len(results.boxes) == 0:
        return "Move Forward"

    for box in results.boxes:
        xy = np.array(box.xyxy[0].cpu()) if hasattr(box.xyxy[0], "cpu") else np.array(box.xyxy[0])
        conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
        if conf < MIN_CONF:
            continue
        x1, y1, x2, y2 = xy[:4]
        cx = (x1 + x2) / 2.0
        if cx < frame_width / 3.0:
            zone["left"] = True
        elif cx < 2 * frame_width / 3.0:
            zone["center"] = True
        else:
            zone["right"] = True

    if not zone["center"]:
        return "Move Forward"
    left_blocked = zone["left"]
    right_blocked = zone["right"]
    if not left_blocked and right_blocked:
        return "Rotate Left"
    if not right_blocked and left_blocked:
        return "Rotate Right"
    if not left_blocked and not right_blocked:
        return "Rotate Left"
    return "Stop"

def annotate_frame(results, frame):
    """Draw detections on frame."""
    try:
        plotted = results.plot()
        if plotted is not None:
            return cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
    except Exception:
        pass
    return frame

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/infer", methods=["POST"])
def infer():
    if "frame" not in request.files:
        return jsonify({"error": "no frame"}), 400

    file = request.files["frame"].read()
    nparr = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    height, width = frame.shape[:2]
    results = model(frame, imgsz=640, verbose=False)[0]
    direction = decide_direction_from_results(results, width)

    annotated = annotate_frame(results, frame)
    cv2.putText(annotated, f"Direction: {direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    _, jpeg = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

    return jsonify({"direction": direction, "image": b64})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
