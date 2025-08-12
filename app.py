import os
import time
import base64
import io
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__, template_folder="templates")

# load local model
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
model = YOLO(MODEL_PATH)

# inference config
MIN_CONF = float(os.environ.get("MIN_CONF", 0.25))

def decide_direction_from_results(results, frame_width):
    """
    Decide direction based on detections:
      - if center free -> Move Forward
      - else if left free -> Rotate Left
      - else if right free -> Rotate Right
      - else Stop
    """
    zone = {"left": False, "center": False, "right": False}
    if results is None or len(results.boxes) == 0:
        return "Move Forward"

    for box in results.boxes:
        # box.xyxy[0] is tensor-like; convert to numpy
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
    """Return annotated image (BGR numpy) using ultralytics plotting if possible."""
    try:
        plotted = results.plot()
        if plotted is not None:
            # results.plot() returns an RGB image usually; ensure BGR for OpenCV
            if plotted.shape[2] == 3:
                # convert RGB->BGR
                plotted_bgr = cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR)
                return plotted_bgr
            return plotted
    except Exception:
        pass

    # fallback: draw boxes manually
    out = frame.copy()
    try:
        for box in results.boxes:
            xy = np.array(box.xyxy[0].cpu()) if hasattr(box.xyxy[0], "cpu") else np.array(box.xyxy[0])
            x1, y1, x2, y2 = map(int, xy[:4])
            conf = float(box.conf[0]) if hasattr(box, "conf") else 1.0
            label = f"{conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(out, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    except Exception:
        pass
    return out

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/infer", methods=["POST"])
def infer():
    """
    Expects a multipart/form-data with key 'frame' (an image file from the browser).
    Returns JSON: { direction: str, image: base64_jpeg }
    """
    if "frame" not in request.files:
        return jsonify({"error": "no frame file"}), 400

    file = request.files["frame"]
    in_memory = file.read()
    # decode to numpy array
    nparr = np.frombuffer(in_memory, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    height, width = frame.shape[:2]

    # run YOLO inference
    try:
        results = model(frame, imgsz=640, verbose=False)[0]
    except Exception as e:
        # on failure, return original frame and unknown direction
        print("YOLO error:", e)
        _, jpeg = cv2.imencode(".jpg", frame)
        b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")
        return jsonify({"direction": "Error", "image": b64})

    # decide direction
    direction = decide_direction_from_results(results, width)

    # annotate frame
    annotated = annotate_frame(results, frame)
    # put direction text
    cv2.putText(annotated, f"Direction: {direction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # encode to JPEG and base64
    _, jpeg = cv2.imencode(".jpg", annotated)
    b64 = base64.b64encode(jpeg.tobytes()).decode("utf-8")

    return jsonify({"direction": direction, "image": b64})

if __name__ == "__main__":
    # for local dev use the flask server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
