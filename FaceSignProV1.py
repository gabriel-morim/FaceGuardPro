import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp


class FaceSignPro:
    def __init__(self):
        self.app_name = "FaceSign Pro v1.0"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.focal_length = 500
        self.known_face_width = 15
        self.face_count = 0
        self.start_time = datetime.now()
        self.credits1 = "Made by: "
        self.credits2 = "Gabriel Morim and Tiago Baganha"
        
        # Sign language detection setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.sign_mode = False
        self.last_gesture = ""
        self.gesture_buffer = []
        
        # Simple ASL gesture mapping (example gestures)
        self.gesture_map = {
            'HELLO': lambda landmarks: self.check_hello_gesture(landmarks),
            'THANK YOU': lambda landmarks: self.check_thank_you_gesture(landmarks),
            'YES': lambda landmarks: self.check_yes_gesture(landmarks),
            'NO': lambda landmarks: self.check_no_gesture(landmarks)
        }

    def check_hello_gesture(self, landmarks):
        # Simplified hello gesture detection (open palm)
        return all(landmarks[i].y < landmarks[0].y for i in [8, 12, 16, 20])
    
    def check_thank_you_gesture(self, landmarks):
        # Simplified thank you gesture
        return landmarks[8].y < landmarks[5].y and landmarks[12].y < landmarks[9].y
    
    def check_yes_gesture(self, landmarks):
        # Simplified yes gesture (thumbs up)
        return landmarks[4].y < landmarks[3].y and landmarks[8].y > landmarks[5].y
    
    def check_no_gesture(self, landmarks):
        # Simplified no gesture (closed fist)
        return all(landmarks[i].y > landmarks[0].y for i in [8, 12, 16, 20])

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
                        if len(self.gesture_buffer) > 10:  # Stability check
                            if all(g == gesture_name for g in self.gesture_buffer[-5:]):
                                self.last_gesture = gesture_name
                        break

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
        
        # Sign language mode status and detected gesture
        mode_text = "Sign Mode: ON" if self.sign_mode else "Sign Mode: OFF (Press 'S')"
        cv2.putText(frame, mode_text, 
                   (10, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.sign_mode and self.last_gesture:
            cv2.putText(frame, f"Detected: {self.last_gesture}", 
                       (width//2 - 100, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            self.face_count = len(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                distance = self.calculate_distance(w)
                cv2.putText(frame, f'Distance: {distance:.2f} cm', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
            # Sign language detection when enabled
            if self.sign_mode:
                self.detect_gesture(frame)
            
            self.draw_info_panel(frame)
            cv2.imshow('FaceGuard Pro', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.sign_mode = not self.sign_mode
                self.last_gesture = ""
                self.gesture_buffer = []
                
        cap.release()
        cv2.destroyAllWindows()

    def calculate_distance(self, face_width_pixels):
        return (self.known_face_width * self.focal_length) / face_width_pixels

if __name__ == "__main__":
    app = FaceSignPro()
    app.run()