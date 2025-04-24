🎧 Anecdote: Where Sound Meets Insight

Real-Time Object Detection & Distance Estimation for the Visually Impaired

Anecdote is an assistive AI-driven web application that enhances accessibility for visually impaired users. It combines real-time object detection, distance estimation, and multilingual audio feedback using cutting-edge technologies like YOLOv8, OpenCV, and Flask.

🧠 Overview

Anecdote captures live video using your webcam, detects objects, estimates their distance from the camera, and speaks out what it sees, making it a powerful tool for the visually impaired.

🚀 Features

🔍 Real-Time Object Detection: Powered by YOLOv8 for fast and accurate identification.
📏 Distance Estimation: Calculates object proximity using OpenCV.
🗣️ Multilingual Audio Feedback: Supports English, Hindi, French, Spanish, and German via Text-to-Speech.
🌐 Web-Based Interface: Works seamlessly in your browser.
🌙 Night Vision Modes: Includes CLAHE, gamma correction, and histogram equalization.
🧠 Edge Detection Modes: Supports Canny, Sobel, and LoG edge enhancement for experimental visualization.
📄 PDF Report: Export the latest detection summary with distances and modes.
🛠️ Technology Stack

Backend: Flask
Frontend: HTML/CSS, JavaScript (served via Flask)
Detection Model: YOLOv8
Image Processing: OpenCV
Text-to-Speech: gTTS (Google Text-to-Speech)
PDF Generation: FPDF
Camera Feed: OpenCV VideoCapture

📦 Installation

Install Dependencies
pip install -r requirements.txt
Download/Place YOLOv8 Model Weights
Place your YOLO model file (e.g., best.pt) at:
/Users/vibhutibhardwaj/runs/detect/train4/weights/best.pt
▶️ Running the App

python app.py
Open your browser and go to:
http://127.0.0.1:5000

🌐 Routes

/start_feed: Start the normal webcam feed
/video_feed: Video stream with object detection (normal mode)
/video_feed_night?mode=clahe|gamma|histogram: Night vision enhancement
/video_feed_edge?mode=canny|sobel|log: Edge detection feed
/generate_pdf: Download a PDF report of the last detected objects

📄 PDF Report

The /generate_pdf endpoint allows users to download a report of the latest detected objects, their distances, and the mode used (normal/night/edge).

📸 Sample Output

Object	Distance (cm)	Mode
Chair	120.5	normal
Person	75.0	clahe
Cup	90.0	canny
⚙️ Configuration

FOCAL_LENGTH and KNOWN_WIDTH are customizable for your camera and real object sizes.
Uses gTTS and afplay (macOS) for audio. You can replace afplay with mpg321 for Linux or playsound on Windows.
🙋‍♀️ Future Improvements

Web-based language selection
Enhanced PDF formatting
Object prioritization by user preference
Mobile-friendly interface
