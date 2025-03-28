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

# Get the width and height of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the areas of interest (AOI) rectangles based on the image map coordinates
aois = [
    (14, 83, 606, 234),   # AOI 1
    (11, 280, 611, 432)   # AOI 2
]

def is_within_aois(x1, y1, x2, y2, aois):
    for index, (ax1, ay1, ax2, ay2) in enumerate(aois):
        if x1 >= ax1 and y1 >= ay1 and x2 <= ax2 and y2 <= ay2:
            return index
    return -1

# Initialize traffic light states
traffic_light_states = ["red", "red"]

# Initialize timer for green light duration
green_light_duration = 10  # in seconds
green_light_timer = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Predict on the frame
    detection_output = model.predict(source=frame, conf=0.25, save=False)

    # Draw the AOI rectangles on the frame
    for idx, (ax1, ay1, ax2, ay2) in enumerate(aois):
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)

    # Initialize object counts
    aoi_counts = [0, 0]

    # Iterate over the detections
    for det in detection_output:
        for x1, y1, x2, y2, conf, cls in det.boxes.data:
            # Check which AOI the detection is within
            aoi_index = is_within_aois(x1, y1, x2, y2, aois)
            if aoi_index != -1:
                aoi_counts[aoi_index] += 1
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Draw the label
                label = f'{cls} {conf:.2f}'
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the object counts on the frame
    cv2.putText(frame, f'Objects in AOI 1: {aoi_counts[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f'Objects in AOI 2: {aoi_counts[1]}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Update traffic light states based on object counts
    if time.time() - green_light_timer >= green_light_duration:
        if aoi_counts[0] > aoi_counts[1]:
            traffic_light_states = ["green", "red"]
        elif aoi_counts[0] < aoi_counts[1]:
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

    # Display the frame with detections, object counts, and traffic light states
    cv2.imshow('YOLOv8 Live Detection', frame)

    # Send the counted object output to a specific link
    try:
        response = requests.post("http://your-url-here.com", data={'aoi1_count': aoi_counts[0], 'aoi2_count': aoi_counts[1], 
                                                                   'traffic_light_aoi1': traffic_light_states[0]})
        print("Response from server:", response.text)
    except Exception as e:
        print(f"Error sending data to server: {e}")

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
