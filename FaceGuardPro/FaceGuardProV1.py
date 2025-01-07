import cv2
import numpy as np

class FaceGuardPro:
    def __init__(self):
        self.app_name = "FaceGuard Pro v1.0"
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.focal_length = 500
        self.known_face_width = 15  # cm

    def calculate_distance(self, face_width_pixels):
        return (self.known_face_width * self.focal_length) / face_width_pixels

    def run(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                distance = self.calculate_distance(w)
                cv2.putText(frame, f'Distance: {distance:.2f} cm', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, self.app_name, 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            cv2.imshow('FaceGuard Pro', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = FaceGuardPro()
    app.run()
