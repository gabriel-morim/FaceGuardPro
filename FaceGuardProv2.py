import cv2
import numpy as np
from datetime import datetime

class FaceGuardPro:
    def __init__(self):
        self.app_name = "FaceGuard Pro v2.0"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.focal_length = 500
        self.known_face_width = 15
        self.face_count = 0
        self.start_time = datetime.now()
        self.credits1 = "Made by:"
        self.credits2 = "Gabriel Morim & Tiago Baganha"

    def draw_info_panel(self, frame):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # Top panel
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        
        # Bottom panel
        cv2.rectangle(overlay, (0, height-40), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # App info
        cv2.putText(frame, self.app_name, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f'Faces: {self.face_count}', 
                   (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Runtime and credits
        runtime = str(datetime.now() - self.start_time).split(".")[0]
        cv2.putText(frame, f'Runtime: {runtime}', 
                   (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Credits at bottom corners
        cv2.putText(frame, self.credits1, 
                   (width//2 - 100, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, self.credits2, 
                   (width//2 + 20, height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def detect_and_draw_eyes(self, frame, x, y, w, h):
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), 
                         (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

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
                
                self.detect_and_draw_eyes(frame, x, y, w, h)
                
                center_x = x + w//2
                center_y = y + h//2
                cv2.circle(frame, (center_x, center_y), 2, (0, 255, 0), -1)
                cv2.line(frame, (center_x - 10, center_y), 
                        (center_x + 10, center_y), (0, 255, 0), 1)
                cv2.line(frame, (center_x, center_y - 10), 
                        (center_x, center_y + 10), (0, 255, 0), 1)
            
            self.draw_info_panel(frame)
            cv2.imshow('FaceGuard Pro', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def calculate_distance(self, face_width_pixels):
        return (self.known_face_width * self.focal_length) / face_width_pixels

if __name__ == "__main__":
    app = FaceGuardPro()
    app.run()