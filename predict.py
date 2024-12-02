
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/obb/n/Ro-YOLOv8-PKI/weights/best.pt')
    results = model.predict('ultralytics/assets/test.jpg', save=True, line_width=6)