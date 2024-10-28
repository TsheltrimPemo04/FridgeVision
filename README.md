# Fridge Vision
Fridge Vision is a deep learning and computer vision-based project aimed at automating household refrigerator management. It addresses food wastage by providing real-time inventory tracking through a web app. The system ensures effective grocery planning and food management, helping reduce waste and promote sustainable living.

### Model Used
This project employs YOLOv5 as the core model for object detection. After evaluating YOLOv3, YOLOv5, and YOLOv8, YOLOv5 was chosen due to its balance between accuracy, speed, and computational efficiency. YOLOv5 demonstrated excellent detection in cluttered environments and under various lighting conditions, making it ideal for refrigerator management.

<img width="731" alt="Screenshot 2024-10-28 at 11 50 24 AM" src="https://github.com/user-attachments/assets/04477005-a539-41fb-86b3-554c68b2c526">

### Dataset
The training dataset was sourced from Kaggle, containing various refrigerator items (e.g., milk, bread, eggs). Images were resized to 416x416 pixels for compatibility with YOLO and augmented to improve performance under different lighting and occlusion conditions.

