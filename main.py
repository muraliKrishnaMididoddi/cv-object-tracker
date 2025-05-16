import cv2
import torch
import numpy as np
from filterpy.kalman import KalmanFilter

# Load YOLOv5 nano model (fastest)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.conf = 0.4

# Track each object by ID
kalman_filters = {}

def create_kalman(cx, cy):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([cx, cy, 0, 0])  # initial state
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= 1000
    kf.R *= 5
    kf.Q = np.eye(4)
    return kf

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, (row['xmin'], row['ymin'], row['xmax'], row['ymax']))
        label = row['name']
        conf = row['confidence']

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        obj_id = f"{label}_{x1}_{y1}"

        if obj_id not in kalman_filters:
            kalman_filters[obj_id] = create_kalman(cx, cy)

        kf = kalman_filters[obj_id]
        kf.predict()
        kf.update([cx, cy])
        px, py = int(kf.x[0]), int(kf.x[1])

        # Draw detection box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw current centroid
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # Draw predicted point
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
        cv2.line(frame, (cx, cy), (px, py), (255, 0, 0), 2)

    cv2.imshow("Fast Object Detection + Motion Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

