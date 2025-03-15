"""Fine tuning YOLO to detect football players, the ball, as well as referees."""

from ultralytics import YOLO

model = YOLO('yolov8l.pt')

dataset = '/Users/gabriel/Documents/GitHub/EAFC-ML-Remaster/data/pretraining_dataset/data.yaml'

model.train(data=dataset, epochs=20)
print("Success!")