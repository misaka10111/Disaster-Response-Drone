import torch
import cv2

class PerceptionModule:
    def __init__(self, model_path="yolov5s.pt"):
        print("[Perception] Loading YOLO model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = 0.45   # confidence threshold
        self.model.iou = 0.45    # NMS threshold

    def detect(self, frame):
        # YOLO expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self.model(rgb_frame)

        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            class_name = self.model.names[int(cls)]

            if class_name in ["person", "fire", "debris"]:
                detections.append({
                    "class": class_name,
                    "confidence": float(conf),
                    "bbox": [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])]
                })

        return {"detections": detections}
