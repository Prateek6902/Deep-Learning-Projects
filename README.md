# APS - Suspicious Activity Detection

**APS (Automated Proctoring System)** is a real-time suspicious activity detection system designed for video proctoring using computer vision and machine learning. This project leverages MediaPipe for face tracking and pose estimation, and provides a sleek, responsive web interface for analysis and visualization.

## 🚀 Features

- 🎥 Upload and analyze recorded exam videos
- 🧠 Head pose estimation to detect side glancing or looking down
- 🤝 Multiple face detection to catch collaboration
- ❌ Face absence detection for abnormal screen presence
- 📊 Visualize results in a stylish dashboard with graphs
- 🌐 Web app with Flask backend and responsive frontend

## 🛠️ Tech Stack

- **Frontend**: HTML, CSS (`style.css`, `styles.css`), JavaScript
- **Backend**: Python Flask
- **Computer Vision**: OpenCV, MediaPipe
- **Visualization**: Chart.js
- **Libraries**: NumPy, datetime, os

## 📁 Project Structure

├── app.py # Flask backend and video analysis logic
├── index.html # Main webpage interface
├── style.css # UI styling (main)
├── styles.css # Additional UI styling and result formatting
├── script.js # JavaScript to handle form submission and chart updates
├── static/ # Folder where uploaded videos are saved


## 🧠 How It Works

1. A user uploads a video through the web interface.
2. Flask handles the file upload and passes it to `analyze_video()` function.
3. MediaPipe FaceMesh tracks facial landmarks.
4. The system detects:
   - Unusual head movement (left, right, down)
   - Multiple faces
   - No face present
5. Suspicious events are logged with timestamps and severity.
6. JSON results are returned and visualized in the browser.

### Sample Detection Output (JSON):
```json
{
  "total_time": 60,
  "suspicious_count": 3,
  "status": "Suspicious",
  "activities": [
    {
      "timestamp": 5,
      "warning": "Looking Left/Right - Potential side glancing",
      "severity": "high",
      "pose": {"x": 0.0, "y": 15.6}
    },
    ...
  ]
}

##" 💻 Run Locally
Requirements
Python 3.7+
pip "

🤝 Acknowledgements
MediaPipe
OpenCV
Flask
Font Awesome
Chart.js

🧑‍💻 This is an original project developed by me for academic/research/personal purposes.
Feel Free to use and modify as per your suit.





