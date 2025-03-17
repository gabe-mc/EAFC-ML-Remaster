"""Tracks the players with a bounding box representation using YOLO"""

import cv2
from ultralytics import YOLO

class PlayerTracker:
    """
    This class creates a real-time stream that tracks football players.

    Attributes:
    path_to_model: the path to the .pt model to be loaded into YOLO
    model: The loaded YOLO model
    """
    
    def __init__(self, path_to_model: str)-> None:
        self.model = YOLO(path_to_model)

    def run_player_tracker(self)-> None:

        video_path = "/Users/gabriel/Documents/GitHub/EAFC-ML-Remaster/data/video_clips/arsenal_chelsea_EAFC_4sec.mp4"
        cap = cv2.VideoCapture(video_path)
        # cap = cv2.VideoCapture(0) # Change to use some screencapture/webcam or something

        if not cap.isOpened():
            print("ERROR: Video stream not opened properly.")
            exit()
    
        while True:
            succ, frame = cap.read()
            if not succ:
                print("ERROR: Could not read frame.")
                break

            results = self.model.predict(frame) # Run single frame inference w/ YOLO

            # # Draw bounding boxes on the frame (REMOVE THIS WHEN WE ACTUALL RUN THE SCRIPT)
            # for result in results:
            #     for box in result.boxes:
            #         x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            #         conf = box.conf[0].item()  # Confidence score
            #         cls = int(box.cls[0].item())  # Class ID

            #         # Draw rectangle and label
            #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #         cv2.putText(frame, f"Class {cls} ({conf:.2f})", (x1, y1 - 10),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("YOLO Real-Time Detection", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # End process
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pt = PlayerTracker("models/yolo_v5_pretrained.pt")
    pt.run_player_tracker()
