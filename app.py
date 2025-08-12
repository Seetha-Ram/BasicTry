import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Threshold for detecting an "obstacle"
OBSTACLE_THRESHOLD = 1500  # pixel area
ZONE_COUNT = 3  # left, center, right

def detect_path(frame):
    """Detect large objects and decide a path."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = frame.shape[:2]
    zone_width = width // ZONE_COUNT
    zones_blocked = [False] * ZONE_COUNT

    for c in contours:
        area = cv2.contourArea(c)
        if area > OBSTACLE_THRESHOLD:
            x, y, w, h = cv2.boundingRect(c)
            zone_index = min(x // zone_width, ZONE_COUNT - 1)
            zones_blocked[zone_index] = True
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Decide direction
    if not zones_blocked[1]:  # center clear
        direction = "Move Forward"
    elif not zones_blocked[0]:
        direction = "Move Left"
    elif not zones_blocked[2]:
        direction = "Move Right"
    else:
        direction = "Stop"

    cv2.putText(frame, f"Direction: {direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return frame, direction

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/infer", methods=["POST"])
def infer():
    if "frame" not in request.files:
        return jsonify({"error": "no frame"}), 400

    file_bytes = request.files["frame"].read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    processed_frame, direction = detect_path(frame)

    _, jpeg = cv2.imencode(".jpg", processed_frame)
    b64_image = base64.b64encode(jpeg).decode("utf-8")

    return jsonify({"direction": direction, "image": b64_image})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
