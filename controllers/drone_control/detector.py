from ultralytics import YOLO

# High-level colors for semantic classes
CATEGORY_COLORS = {
    "person":   (255, 60, 60),
    "vehicle":  (70, 130, 180),
    "debris":   (184, 134, 11),
    "machine":  (255, 215, 0),
    "fire":     (255, 140, 0),
}

# YOLO → high-level mapping
CATEGORY_MAP = {
    "person": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "excavator": "machine",
    "fire": "fire",
}


def map_yolo_class(name: str) -> str:
    return CATEGORY_MAP.get(name, "debris")


# Load YOLO model globally (so it loads only once)
_model = YOLO("best.pt")


def detect_objects(img_rgb):
    """
    Run YOLO detection.
    Returns the YOLO result object.
    """
    return _model(img_rgb)[0]
