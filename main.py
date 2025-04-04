import cv2
import time
from datetime import datetime
from face_detector import FaceDetector
from tracker import KalmanFaceTracker
from database import FaceDatabase

class FaceTrackingApp:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.trackers = {}  # face_id: KalmanFaceTracker
        self.database = FaceDatabase()
        self.next_face_id = 1
        
    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                    
                # Detect faces
                faces = self.face_detector.detect_faces(frame)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                
                # Process each detected face
                for face in faces:
                    x1, y1, x2, y2 = face['box']
                    width = x2 - x1
                    height = y2 - y1
                    confidence = face['confidence']
                    
                    # Find the best matching existing face
                    best_match_id = None
                    best_iou = 0.5  # Minimum IoU threshold
                    
                    for face_id, tracker in self.trackers.items():
                        # Get predicted position from tracker
                        tx, ty, tw, th = tracker.kalman.statePost[:4]
                        
                        # Calculate IoU (Intersection over Union)
                        intersection = max(0, min(x2, tx+tw) - max(x1, tx)) * max(0, min(y2, ty+th) - max(y1, ty))
                        union = width * height + tw * th - intersection
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_match_id = face_id
                    
                    if best_match_id:
                        # Update existing tracker
                        tracker = self.trackers[best_match_id]
                        tracker.update([x1, y1, width, height])
                        
                        # Add to database with timestamp
                        face_image = frame[y1:y2, x1:x2].copy()
                        self.database.update_face(best_match_id, face_image, confidence, 
                                                 x1, y1, width, height, current_time)
                        
                        # Draw rectangle and ID
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {best_match_id}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # New face detected
                        face_id = str(self.next_face_id)
                        self.next_face_id += 1
                        
                        # Create new tracker
                        tracker = KalmanFaceTracker()
                        tracker.update([x1, y1, width, height])
                        self.trackers[face_id] = tracker
                        
                        # Add to database with timestamp
                        face_image = frame[y1:y2, x1:x2].copy()
                        self.database.add_face(face_image, confidence, 
                                             x1, y1, width, height, current_time)
                        
                        # Draw rectangle and ID
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"New ID: {face_id}", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow('Face Tracking', frame)
                
                # Exit on 'q' key or window close
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Face Tracking', cv2.WND_PROP_VISIBLE) < 1:
                    print("User requested to quit")
                    break
                    
        except KeyboardInterrupt:
            print("\nProgram interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Release resources in finally block to ensure they're always released
            print("\nReleasing resources...")
            if 'cap' in locals() and isinstance(cap, cv2.VideoCapture):
                if cap.isOpened():
                    cap.release()
                    print("Camera released")
            cv2.destroyAllWindows()
            print("OpenCV windows closed")
            if hasattr(self, 'database'):
                self.database.close()
                print("Database connection closed")
            print("Cleanup complete")

if __name__ == "__main__":
    app = FaceTrackingApp()
    try:
        app.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)