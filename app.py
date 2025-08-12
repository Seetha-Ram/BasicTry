from flask import Flask, Response, render_template_string
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

last_direction = None
last_update_time = time.time()
update_interval = 3  # seconds

# Frame generator for video streaming
def gen_frames():
    global last_direction, last_update_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        results = model(frame)
        annotated_frame = results[0].plot()

        zone_status = {"left": False, "center": False, "right": False}

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                box_center_x = (x1 + x2) / 2

                if box_center_x < width / 3:
                    zone_status["left"] = True
                elif box_center_x < 2 * width / 3:
                    zone_status["center"] = True
                else:
                    zone_status["right"] = True

        # Decide direction
        if not zone_status["center"]:
            direction = "Move Forward"
        elif not zone_status["left"]:
            direction = "Move Left"
        elif not zone_status["right"]:
            direction = "Move Right"
        else:
            direction = "Stop"

        # Only update direction periodically or when it changes
        current_time = time.time()
        if direction != last_direction or (current_time - last_update_time > update_interval):
            last_direction = direction
            last_update_time = current_time

        # Display direction on frame
        cv2.putText(annotated_frame, f"Direction: {direction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for homepage
@app.route('/')
def index():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Path Planning Assistant</title>
        <style>
            body {
                background-color: #f4f4f4;
                font-family: Arial, sans-serif;
                text-align: center;
                padding: 20px;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            .video-container {
                display: inline-block;
                border: 4px solid #333;
                box-shadow: 0 0 15px rgba(0,0,0,0.3);
            }
            img {
                width: 800px;
                height: auto;
            }
            footer {
                margin-top: 30px;
                color: #777;
            }
        </style>
    </head>
    <body>
        <h1>Live Navigation Feed</h1>
        <div class="video-container">
            <img src="{{ url_for('video_feed') }}" alt="Live Feed">
        </div>
        <footer>
            <p>Powered by YOLOv8 and Flask | Real-time Object Detection & Navigation</p>
        </footer>
    </body>
    </html>
    """
    return render_template_string(html)

# Route for video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
