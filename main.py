from ultralytics import YOLO
import numpy as np
import cv2
import requests
import time

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt", "v8")

# Open a connection to the webcam (source=0, typically for built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize timer for traffic light control
green_light_duration = 10  # in seconds
green_light_timer = time.time()
traffic_light_states = ["red", "red"]

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Predict on the frame
    detection_output = model.predict(source=frame, conf=0.25, save=False)

    # Initialize object counts
    object_counts = [0, 0]

    # Draw rectangles around detected objects and count them
    for det in detection_output:
        for box in det.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            object_counts[0] += 1  # Increment count for all detected objects

    # Update traffic light states based on object counts
    if time.time() - green_light_timer >= green_light_duration:
        if object_counts[0] > object_counts[1]:
            traffic_light_states = ["green", "red"]
        elif object_counts[0] < object_counts[1]:
            traffic_light_states = ["red", "green"]
        else:
            traffic_light_states = ["green", "green"]
        
        # Reset the green light timer
        green_light_timer = time.time()

    # Display traffic light states
    cv2.putText(frame, f'Traffic Light AOI 1: {traffic_light_states[0]}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Traffic Light AOI 2: {traffic_light_states[1]}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display countdown timer after green signal
    if "green" in traffic_light_states:
        countdown_time = green_light_duration - (time.time() - green_light_timer)
        cv2.putText(frame, f'Green light countdown: {countdown_time:.1f} s', (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with detections and traffic light states
    cv2.imshow('YOLOv8 Live Detection', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
