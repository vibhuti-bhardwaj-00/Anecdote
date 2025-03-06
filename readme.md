Anecdote: Where Sound Meets Insight
 Real-Time Object Detection & Distance Estimation with YOLO & OpenCV

Overview
Anecdote is an advanced assistive technology designed to enhance accessibility for visually impaired users by combining real-time object detection, distance estimation, and adaptive audio feedback.

Features:
* Object Detection – Utilizes YOLOv5 for accurate and efficient real-time detection.
* Distance Estimation – Integrates OpenCV to calculate object proximity dynamically.
* Multilingual Audio Feedback – Converts detected objects and their distances into spoken feedback using Text-to-Speech (TTS).
* Web-Based Interface – Seamless deployment for real-time interaction via a browser.

Technology Stack
YOLOv8 – Deep learning-based object detection.
OpenCV – Image processing and distance estimation.
Flask / Django (if applicable) – Backend framework for serving model predictions.
JavaScript & HTML/CSS – Frontend interface for live object tracking.
Text-to-Speech (TTS) API – Converts detections into real-time voice output.
How It Works
Live Webcam Feed – Captures real-time video.
YOLOv8 Detection – Identifies objects in the frame.
Distance Calculation – Estimates distance based on object size.
Audio Feedback – Announces detected objects and their distances.