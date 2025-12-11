import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from detector import detect_objects, CATEGORY_COLORS, map_yolo_class
from debris import detect_debris_unsupervised
from drawing import draw_pin, draw_legend, draw_grid
from pathfinding import move_robot_toward_multiple_targets
from footprint import animate_uav_path


def uav_topdown_disaster_map(img_path, meters_per_pixel=0.12, cell_size_m=5.0):
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    result = detect_objects(img_rgb)
    debris_mask = detect_debris_unsupervised(img_rgb)

    # 1) Initialize pure white base image
    base = np.ones((H, W, 3), float)  # All 1s → pure white (RGB = 1,1,1)

    # 2) Calculate debris probability map (with smoothing)
    debris_prob = cv2.GaussianBlur(debris_mask.astype(float), (45, 45), 0)
    if debris_prob.max() > 0:
        debris_prob /= debris_prob.max()

    # 3) Yellow color for debris areas
    debris_color = np.array([225, 195, 110]) / 255.0

    # 4) Blend white base + yellow debris
    colored = base.copy()
    for c in range(3):  # 3 color channels
        colored[..., c] = (
                base[..., c] * (1 - debris_prob * 0.8)+
                debris_color[c] * (debris_prob * 0.9)
        )

    width_m = W * meters_per_pixel
    height_m = H * meters_per_pixel

    persons = []
    objects = []

    for box in result.boxes:
        name = result.names[int(box.cls)]
        mapped = map_yolo_class(name)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) / 2 * meters_per_pixel
        cy = (y1 + y2) / 2 * meters_per_pixel

        objects.append((mapped, cx, cy))
        if mapped == "person":
            persons.append((cx, cy))

    if not persons:
        print(" No survivors detected.")
        return

    uav_path = move_robot_toward_multiple_targets((0, 0), persons)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(base, extent=[0, width_m, 0, height_m], origin="lower")

    draw_grid(ax, width_m, height_m, cell_size_m)

    for mapped, cx, cy in objects:
        color = np.array(CATEGORY_COLORS[mapped]) / 255.0
        draw_pin(ax, (cx, cy), color, mapped.capitalize(), size=cell_size_m * 0.22)

    ani = animate_uav_path(ax, uav_path, interval=70)

    draw_legend(ax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("UAV Top-down SLAM Map + Animated Path", fontsize=16)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    uav_topdown_disaster_map(
        r"E:\Google\intelligent robotics\pythonProject\Mapping\Disaster_economy_PPPs.jpg",
        meters_per_pixel=0.1,
        cell_size_m=5.0
    )