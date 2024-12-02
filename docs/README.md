# Rice Leaf Blast Disease Detection with Ro-YOLOv8-PKI

This repository contains a YOLOv8-based model for detecting rice leaf blast disease in agricultural fields. The model is trained to detect the disease’s symptoms on rice leaves, providing an efficient way for early diagnosis and crop protection.

## Example Code

```python
from ultralytics import YOLO

if __name__ == '__main__':
    # Load pre-trained YOLOv8 model
    model = YOLO('runs/obb/n/Ro-YOLOv8-PKI/weights/best.pt')
    
    # Make predictions on the input image and save results
    results = model.predict('ultralytics/assets/test.jpg', save=True, line_width=6)
