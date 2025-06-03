from flask import Flask, render_template, request, jsonify, url_for
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import os

app = Flask(__name__, 
    template_folder='.',    # Use current directory for templates
    static_folder='.',      # Use current directory for static files
    static_url_path=''      # This ensures static files are served from root
)
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def analyze_video(video_path):
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    suspicious_activities = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    faces_detected_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:  # Analyze every 30 frames
            # Convert BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = face_mesh.process(image)
            
            if results.multi_face_landmarks:
                faces_detected_count += 1
                
                # Get head pose
                face_3d = []
                face_2d = []
                face_ids = [33, 263, 1, 61, 291, 199]
                
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in face_ids:
                            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])
                
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)
                
                # Calculate head pose
                focal_length = frame.shape[1]
                cam_matrix = np.array([
                    [focal_length, 0, frame.shape[0] / 2],
                    [0, focal_length, frame.shape[1] / 2],
                    [0, 0, 1]
                ])
                
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                
                x = angles[0] * 360
                y = angles[1] * 360
                
                timestamp = frame_count // 30
                
                # Detect suspicious head movements
                if y < -10 or y > 10:
                    suspicious_activities.append({
                        'timestamp': timestamp,
                        'warning': 'Looking Left/Right - Potential side glancing',
                        'severity': 'high',
                        'pose': {'x': float(x), 'y': float(y)}
                    })
                elif x < -10:
                    suspicious_activities.append({
                        'timestamp': timestamp,
                        'warning': 'Looking Down - Potential cheating',
                        'severity': 'high',
                        'pose': {'x': float(x), 'y': float(y)}
                    })
                
                if len(results.multi_face_landmarks) > 1:
                    suspicious_activities.append({
                        'timestamp': timestamp,
                        'warning': f'Multiple faces detected ({len(results.multi_face_landmarks)}) - Potential collaboration',
                        'severity': 'high'
                    })
            else:
                if faces_detected_count == 0 or frame_count > 90:
                    suspicious_activities.append({
                        'timestamp': timestamp,
                        'warning': 'No face detected - Student might be absent',
                        'severity': 'high'
                    })
    
    cap.release()
    face_mesh.close()
    
    # Add summary with face detection stats
    analysis_summary = {
        'total_time': total_frames // 30,
        'suspicious_count': len(suspicious_activities),
        'status': 'Suspicious' if len(suspicious_activities) > 0 else 'Normal',
        'activities': suspicious_activities,
        'faces_detected_frames': faces_detected_count,
        'total_analyzed_frames': frame_count // 30
    }
    
    return analysis_summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(filepath)
    
    results = analyze_video(filepath)
    return jsonify({'results': results})

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
