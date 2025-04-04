import os
import cv2
from datetime import datetime

class FileManager:
    def __init__(self, base_folder="captured_faces"):
        self.base_folder = base_folder
        os.makedirs(self.base_folder, exist_ok=True)
        
    def save_face_image(self, face_image, face_id=None):
        """Save face image with timestamp and return the file path"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if face_id:
            filename = f"{timestamp}_ID_{face_id}.jpg"
        else:
            filename = f"{timestamp}.jpg"
            
        filepath = os.path.join(self.base_folder, filename)
        cv2.imwrite(filepath, face_image)
        return filepath