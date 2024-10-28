# import torch
# import cv2
# import numpy as np
# import os

# # Download YOLOv5 repository if not already downloaded
# if not os.path.exists('yolov5'):
#     os.system('git clone https://github.com/ultralytics/yolov5')

# # Load YOLOv5 model
# model = torch.hub.load('yolov5', 'custom', path='/Users/tsheltrimpemo/Desktop/realtime/best.pt', source='local')  # Load custom model from local path

# # Initialize webcam
# cap = cv2.VideoCapture(0)  # 0 is the default webcam

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame.")
#         break

#     # Convert frame to RGB (YOLOv5 expects RGB images)
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Perform inference
#     results = model(img_rgb)

#     # Parse results
#     annotated_frame = np.squeeze(results.render())  # Render the results on the frame

#     # Display the resulting frame
#     cv2.imshow('YOLOv5 Webcam', annotated_frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close windows
# cap.release()
# cv2.destroyAllWindows()

import torch
import cv2
import numpy as np
import os

# Download YOLOv5 repository if not already downloaded
if not os.path.exists('yolov5'):
    os.system('git clone https://github.com/ultralytics/yolov5')

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='/Users/tsheltrimpemo/Desktop/realtime/best.pt', source='local')  # Load custom model from local path

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Adjust webcam properties (optional, for better color balance)
cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # Adjust exposure (may vary depending on your webcam)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to RGB (YOLOv5 expects RGB images for inference)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Parse results and render annotations on the frame
    annotated_frame = np.squeeze(results.render())  # Render the results on the frame

    # Convert back to BGR for correct OpenCV display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Webcam', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
