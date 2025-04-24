from flask import Flask, Response, render_template, jsonify, render_template_string, send_file
from flask_cors import CORS
import cv2
import time
import os
from gtts import gTTS
from ultralytics import YOLO 
from flask import request
import numpy as np
from fpdf import FPDF


app = Flask(__name__)
CORS(app)

model = YOLO('/Users/vibhutibhardwaj/runs/detect/train4/weights/best.pt')

KNOWN_WIDTH = 7.0  
FOCAL_LENGTH = 1671.43  

LANGUAGE_MAP = {"English": "en", "Hindi": "hi", "French": "fr", "Spanish": "es", "German": "de"}
SELECTED_LANGUAGE = "English"

camera = None
streaming = False
latest_detections = [
    
]



def estimate_distance(known_width, focal_length, pixel_width):
    return (known_width * focal_length / pixel_width) if pixel_width > 0 else 0

def speak(text, lang="en"):
    try:
        print(f"🔊 Speaking: {text} in {lang}")
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        os.system("afplay output.mp3") 
    except Exception as e:
        print(f"❌ Error in speech synthesis: {e}")
        


def generate_frames(mode="normal"):
    global camera, streaming
    last_spoken_time = time.time()
    min_speak_interval = 3  

    while streaming:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)  
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
                    nearest_object = (class_name, x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}: {distance:.2f} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if nearest_object and (time.time() - last_spoken_time > min_speak_interval):
            class_name, x1, y1, x2, y2 = nearest_object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            cv2.putText(frame, f"Closest: {class_name} ({min_distance:.2f} cm)", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            spoken_text = f"{class_name} detected at {int(min_distance)} centimeters"
            speak(spoken_text, LANGUAGE_MAP[SELECTED_LANGUAGE])
            last_spoken_time = time.time()
            
            latest_detections.append({
            "object": class_name,
            "distance": round(distance, 2),
            "mode": mode
            })
            print("Latest detections updated:", latest_detections)


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_night(mode):
    global camera, streaming
    last_spoken_time = time.time()
    min_speak_interval = 3  

    while streaming:
        success, frame = camera.read()
        if not success:
            break

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if mode == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)
        elif mode == 'gamma':
            gamma = 1.5  
            enhanced_gray = cv2.convertScaleAbs(gray, alpha=1, beta=gamma)
        elif mode == 'histogram':
            enhanced_gray = cv2.equalizeHist(gray)

        enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

        
        results = model(enhanced_frame)
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
                    nearest_object = (class_name, x1, y1, x2, y2)

                
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(enhanced_frame, f"{class_name}: {distance:.2f} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        
        if nearest_object and (time.time() - last_spoken_time > min_speak_interval):
            class_name, x1, y1, x2, y2 = nearest_object
            spoken_text = f"{class_name} detected at {int(min_distance)} centimeters"
            speak(spoken_text, LANGUAGE_MAP[SELECTED_LANGUAGE])
            last_spoken_time = time.time()
            
            latest_detections.append({
            "object": class_name,
            "distance": round(distance, 2),
            "mode": mode
            })

        
        ret, buffer = cv2.imencode('.jpg', enhanced_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_edge(mode='canny'):
    global camera, streaming
    last_spoken_time = time.time()
    min_speak_interval = 3  

    while streaming:
        success, frame = camera.read()
        if not success:
            break

        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

       
        if mode == 'canny':
           
            edges = cv2.Canny(gray, 50, 150)
        elif mode == 'sobel':
            
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(grad_x, grad_y)
            edges = cv2.convertScaleAbs(magnitude)
        elif mode == 'log':
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.absolute(laplacian)
            edges = cv2.convertScaleAbs(laplacian)

       
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        
        results = model(frame)
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
                    nearest_object = (class_name, x1, y1, x2, y2)

               
                cv2.rectangle(edges_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(edges_colored, f"{class_name}: {distance:.2f} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

       
        if nearest_object and (time.time() - last_spoken_time > min_speak_interval):
            class_name, x1, y1, x2, y2 = nearest_object
            spoken_text = f"{class_name} detected at {int(min_distance)} centimeters"
            speak(spoken_text, LANGUAGE_MAP[SELECTED_LANGUAGE])
            last_spoken_time = time.time()

            latest_detections.append({
            "object": class_name,
            "distance": round(distance, 2),
            "mode": mode
            })
            
       
        ret, buffer = cv2.imencode('.jpg', edges_colored)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_feed')
def start_feed():
    global camera, streaming
    if not streaming:
        camera = cv2.VideoCapture(0)  
        streaming = True
    return jsonify({"status": "started"})

@app.route('/stop_feed')
def stop_feed():
    global camera, streaming
    if streaming:
        streaming = False
        if camera is not None:
            camera.release()  
            camera = None
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    global streaming
    if not streaming:
        return jsonify({"error": "Feed not started"}), 400
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_night')
def video_feed_night():
    mode = request.args.get('mode', default='clahe', type=str)  # Get mode from query params
    global streaming
    if not streaming:
        return jsonify({"error": "Feed not started"}), 400
    return Response(generate_frames_night(mode), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_edge')
def video_feed_edge():
    global streaming
    
    mode = request.args.get('mode', default='canny', type=str)

    if not streaming:
        return jsonify({"error": "Feed not started"}), 400
    
    
    return Response(generate_frames_edge(mode), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_edge_detection')
def start_edge_detection():
    global camera, streaming
    if not streaming:
        camera = cv2.VideoCapture(0)  
        streaming = True
    return jsonify({"status": "started"})





@app.route('/generate_pdf')
def generate_pdf():
    
    pdf = FPDF()
    pdf.add_page()

    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Latest Detections", ln=True, align='C')

    
    pdf.ln(10)  

    
    for detection in latest_detections:
        pdf.cell(200, 10, txt=f"Object: {detection['object']}, Distance: {detection['distance']} cm, Mode: {detection['mode']}", ln=True)

    
    pdf_output_path = 'latest_detections.pdf'  
    pdf.output(pdf_output_path)

    
    return send_file(pdf_output_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)


 
 
