import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class FaceSignPro:
    def __init__(self):
        self.app_name = "FaceSign Pro v1.0"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Distance calculation parameters
        self.calibration_distance = 100  # Distance in cm during calibration
        self.known_face_width = 15  # Average face width in cm
        self.focal_length = 500  # Default focal length
        self.is_calibrated = False
        
        # Tracking variables
        self.face_count = 0
        self.start_time = datetime.now()
        self.distance_history = []
        self.smooth_window = 5
        
        # Credits
        self.credits1 = "Made by: "
        self.credits2 = "Gabriel Morim and Tiago Baganha"
        
        # Sign language detection setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        # Sign tracking
        self.sign_mode = False
        self.current_sign = ""
        self.gesture_buffer = []
        self.sign_timeout = 1  # Seconds before allowing new sign detection
        self.last_sign_time = datetime.now()
        
        # Simple ASL gesture mapping
        self.gesture_map = {
            'HELLO': lambda landmarks: self.check_hello_gesture(landmarks),
            'THANK YOU': lambda landmarks: self.check_thank_you_gesture(landmarks),
            'YES': lambda landmarks: self.check_yes_gesture(landmarks),
            'NO': lambda landmarks: self.check_no_gesture(landmarks)
        }

    def check_hello_gesture(self, landmarks):
        return all(landmarks[i].y < landmarks[0].y for i in [8, 12, 16, 20])
    
    def check_thank_you_gesture(self, landmarks):
        return landmarks[8].y < landmarks[5].y and landmarks[12].y < landmarks[9].y
    
    def check_yes_gesture(self, landmarks):
        return landmarks[4].y < landmarks[3].y and landmarks[8].y > landmarks[5].y
    
    def check_no_gesture(self, landmarks):
        return all(landmarks[i].y > landmarks[0].y for i in [8, 12, 16, 20])

    def calibrate_focal_length(self, face_width_pixels):
        """Calibrate focal length using a known distance and face width"""
        self.focal_length = (face_width_pixels * self.calibration_distance) / self.known_face_width
        self.is_calibrated = True
        print(f"Calibrated focal length: {self.focal_length}")

    def calculate_distance(self, face_width_pixels):
        """Calculate distance with improved accuracy and smoothing"""
        if not self.is_calibrated:
            return self.calibration_distance
        
        current_distance = (self.known_face_width * self.focal_length) / face_width_pixels
        
        # Apply smoothing
        self.distance_history.append(current_distance)
        if len(self.distance_history) > self.smooth_window:
            self.distance_history.pop(0)
        
        # Remove outliers
        distances = np.array(self.distance_history)
        mean = np.mean(distances)
        std = np.std(distances)
        valid_distances = distances[abs(distances - mean) < 2 * std]
        
        return np.mean(valid_distances) if len(valid_distances) > 0 else current_distance

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                for gesture_name, check_func in self.gesture_map.items():
                    if check_func(landmarks):
                        self.gesture_buffer.append(gesture_name)
                        if len(self.gesture_buffer) > 10:
                            if all(g == gesture_name for g in self.gesture_buffer[-5:]):
                                time_since_last = (datetime.now() - self.last_sign_time).total_seconds()
                                if time_since_last > self.sign_timeout:
                                    self.current_sign = gesture_name
                                    self.last_sign_time = datetime.now()
                        break
        else:
            # Clear current sign if no hands detected for more than timeout period
            time_since_last = (datetime.now() - self.last_sign_time).total_seconds()
            if time_since_last > self.sign_timeout:
                self.current_sign = ""

    def draw_info_panel(self, frame):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Panels
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, height-80), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Top info
        cv2.putText(frame, self.app_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Faces: {self.face_count}', 
                   (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current Sign (large, centered display)
        if self.sign_mode and self.current_sign:
            sign_text = f"Sign: {self.current_sign}"
            text_size = cv2.getTextSize(sign_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height - 80 - text_size[1]) // 2
            cv2.putText(frame, sign_text, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Bottom info
        runtime = str(datetime.now() - self.start_time).split(".")[0]
        cv2.putText(frame, f'Runtime: {runtime}', 
                   (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Credits
        cv2.putText(frame, self.credits1, 
                   (width//2 - 100, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, self.credits2, 
                   (width//2 + 20, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Sign mode status
        mode_text = "Sign Mode: ON" if self.sign_mode else "Sign Mode: OFF (Press 'S')"
        cv2.putText(frame, mode_text, 
                   (10, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Calibration status
        status = "Calibrated" if self.is_calibrated else "Not Calibrated - Press 'c'"
        cv2.putText(frame, f'Status: {status}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def run(self):
        try:
            print("Initializing camera...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("Cannot open camera")
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            print("Camera initialized successfully")
            print("Controls:")
            print("- Press 'c' to calibrate distance (stand at 100cm)")
            print("- Press 's' to toggle sign language mode")
            print("- Press 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                self.face_count = len(faces)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    distance = self.calculate_distance(w)
                    
                    # Add confidence indicator
                    confidence = "High" if distance < 200 else "Medium" if distance < 400 else "Low"
                    cv2.putText(frame, f'Distance: {distance:.1f} cm ({confidence})', 
                               (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (0, 255, 0), 2)
                
                # Sign language detection when enabled
                if self.sign_mode:
                    self.detect_gesture(frame)
                
                self.draw_info_panel(frame)
                cv2.imshow('FaceSign Pro', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('c') and len(faces) > 0:
                    largest_face = max(faces, key=lambda f: f[2])
                    self.calibrate_focal_length(largest_face[2])
                    print("Calibration complete!")
                elif key == ord('s'):
                    self.sign_mode = not self.sign_mode
                    self.current_sign = ""
                    self.gesture_buffer = []
                    print("Sign mode:", "ON" if self.sign_mode else "OFF")
        
        except Exception as e:
            print(f"Error: {str(e)}")
        
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            print("Application closed")

if __name__ == "__main__":
    try:
        print("Starting FaceSign Pro...")
        app = FaceSignPro()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {str(e)}")