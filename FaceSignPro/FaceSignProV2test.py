import cv2
import numpy as np
from datetime import datetime
import mediapipe as mp

class FaceGuardPro:
    def __init__(self):
        self.app_name = "FaceSign Pro v1.0"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.focal_length = 500
        self.known_face_width = 15
        self.face_count = 0
        self.start_time = datetime.now()
        self.credits1 = "Made by Gabriel Morim"
        self.credits2 = "& Tiago Baganha"
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        
        self.sign_mode = False
        self.last_gesture = ""
        self.gesture_buffer = []
        self.tracking_history = []
        self.max_history = 30
        
        self.gesture_map = {
            'HELLO': lambda landmarks: self.check_hello_gesture(landmarks),
            'THANK YOU': lambda landmarks: self.check_thank_you_gesture(landmarks),
            'YES': lambda landmarks: self.check_yes_gesture(landmarks),
            'NO': lambda landmarks: self.check_no_gesture(landmarks),
            'PEACE': lambda landmarks: self.check_peace_gesture(landmarks),
            'OKAY': lambda landmarks: self.check_okay_gesture(landmarks),
            'WAVE': lambda landmarks: self.check_wave_gesture(landmarks),
            'POINT': lambda landmarks: self.check_point_gesture(landmarks)
        }

    def get_hand_direction(self, current_landmarks, prev_landmarks):
        if not prev_landmarks:
            return "STATIC"
        
        wrist_current = current_landmarks[0]
        wrist_prev = prev_landmarks[0]
        
        dx = wrist_current.x - wrist_prev.x
        dy = wrist_current.y - wrist_prev.y
        
        threshold = 0.02
        if abs(dx) < threshold and abs(dy) < threshold:
            return "STATIC"
        
        if abs(dx) > abs(dy):
            return "RIGHT" if dx > 0 else "LEFT"
        else:
            return "DOWN" if dy > 0 else "UP"

    def check_hello_gesture(self, landmarks):
        return all(landmarks[i].y < landmarks[0].y for i in [8, 12, 16, 20])
    
    def check_thank_you_gesture(self, landmarks):
        return landmarks[8].y < landmarks[5].y and landmarks[12].y < landmarks[9].y
    
    def check_yes_gesture(self, landmarks):
        return landmarks[4].y < landmarks[3].y and landmarks[8].y > landmarks[5].y
    
    def check_no_gesture(self, landmarks):
        return all(landmarks[i].y > landmarks[0].y for i in [8, 12, 16, 20])

    def check_peace_gesture(self, landmarks):
        return (landmarks[8].y < landmarks[5].y and
                landmarks[12].y < landmarks[9].y and
                landmarks[16].y > landmarks[13].y and
                landmarks[20].y > landmarks[17].y)

    def check_okay_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        return distance < 0.1

    def check_wave_gesture(self, landmarks):
        if len(self.tracking_history) < 10:
            return False
        
        directions = [movement["direction"] for movement in self.tracking_history[-10:]]
        has_left = "LEFT" in directions
        has_right = "RIGHT" in directions
        alternating = sum(1 for i in range(len(directions)-1) if directions[i] != directions[i+1])
        
        return has_left and has_right and alternating > 4

    def check_point_gesture(self, landmarks):
        return (landmarks[8].y < landmarks[5].y and
                landmarks[12].y > landmarks[9].y and
                landmarks[16].y > landmarks[13].y and
                landmarks[20].y > landmarks[17].y)

    def detect_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                landmarks = hand_landmarks.landmark
                prev_landmarks = self.tracking_history[-1]["landmarks"] if self.tracking_history else None
                direction = self.get_hand_direction(landmarks, prev_landmarks)
                
                self.tracking_history.append({
                    "landmarks": landmarks,
                    "direction": direction,
                    "timestamp": datetime.now()
                })
                
                if len(self.tracking_history) > self.max_history:
                    self.tracking_history.pop(0)
                
                self.draw_movement_trail(frame)
                
                for gesture_name, check_func in self.gesture_map.items():
                    if check_func(landmarks):
                        self.gesture_buffer.append(gesture_name)
                        if len(self.gesture_buffer) > 10:
                            if all(g == gesture_name for g in self.gesture_buffer[-5:]):
                                self.last_gesture = gesture_name
                        break

    def draw_movement_trail(self, frame):
        if len(self.tracking_history) < 2:
            return
            
        height, width = frame.shape[:2]
        points = []
        
        for record in self.tracking_history:
            wrist = record["landmarks"][0]
            x = int(wrist.x * width)
            y = int(wrist.y * height)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            alpha = i / len(points)
            color = (0, int(255 * alpha), int(255 * (1 - alpha)))
            cv2.line(frame, points[i], points[i + 1], color, 2)

    def draw_info_panel(self, frame):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, height-80), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, self.app_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Faces: {self.face_count}', 
                   (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        runtime = str(datetime.now() - self.start_time).split(".")[0]
        cv2.putText(frame, f'Runtime: {runtime}', 
                   (10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, self.credits1, 
                   (width//2 - 100, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, self.credits2, 
                   (width//2 + 20, height - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        mode_text = "Sign Mode: ON" if self.sign_mode else "Sign Mode: OFF (Press 'S')"
        cv2.putText(frame, mode_text, 
                   (10, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self.sign_mode and self.last_gesture:
            cv2.putText(frame, f"Detected: {self.last_gesture}", 
                       (width//2 - 100, height - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def calculate_distance(self, face_width_pixels):
        return (self.known_face_width * self.focal_length) / face_width_pixels

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            self.face_count = len(faces)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                distance = self.calculate_distance(w)
                cv2.putText(frame, f'Distance: {distance:.2f} cm', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
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

if __name__ == "__main__":
    app = FaceGuardPro()
    app.run()
