from flask import Flask, Response, render_template, jsonify
from flask_cors import CORS
import cv2
import time
import os
from gtts import gTTS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)  


model = YOLO('/Users/vibhutibhardwaj/runs/detect/train4/weights/best.pt')


KNOWN_WIDTH = 7.0  
FOCAL_LENGTH = 1428.57  


LANGUAGE_MAP = {"English": "en", "Hindi": "hi", "French": "fr", "Spanish": "es", "German": "de"}
SELECTED_LANGUAGE = "English"  


camera = None
streaming = False


def estimate_distance(known_width, focal_length, pixel_width):
    return (known_width * focal_length / pixel_width) if pixel_width > 0 else 0


def speak(text, lang="en"):
    try:
        print(f"🔊 Speaking: {text} in {lang}")
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        os.system("afplay output.mp3")  # macOS
    except Exception as e:
        print(f"❌ Error in speech synthesis: {e}")

# Function to process video frames and generate continuous audio
def generate_frames():
    global camera, streaming
    last_spoken_time = time.time()
    min_speak_interval = 3  # Speak once every 3 seconds

    while streaming:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)  # Run YOLO detection
        nearest_object = None
        min_distance = float('inf')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                pixel_width = x2 - x1
                distance = estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, pixel_width)

                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if distance < min_distance and distance > 0:
                    min_distance = distance
                    nearest_object = class_name

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}: {distance:.2f} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Speak nearest object at controlled intervals
        if nearest_object and (time.time() - last_spoken_time > min_speak_interval):
            spoken_text = f"{nearest_object} detected at {int(min_distance)} centimeters"
            speak(spoken_text, LANGUAGE_MAP[SELECTED_LANGUAGE])
            last_spoken_time = time.time()

        # Encode and stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Start and Stop Feed Routes
@app.route('/start_feed')
def start_feed():
    global camera, streaming
    if not streaming:
        camera = cv2.VideoCapture(0)  # Start webcam
        streaming = True
    return jsonify({"status": "started"})

@app.route('/stop_feed')
def stop_feed():
    global camera, streaming
    if streaming:
        streaming = False
        if camera is not None:
            camera.release()  # Release webcam
            camera = None
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    global streaming
    if not streaming:
        return jsonify({"error": "Feed not started"}), 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
