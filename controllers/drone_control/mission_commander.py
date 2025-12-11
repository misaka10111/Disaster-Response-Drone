import json
import os
import cv2
import numpy as np
import time

# 引入项目现有的模块
from detector import detect_objects, map_yolo_class
from pathfinding import move_robot_toward_multiple_targets


class MissionCommander:
    def __init__(self):
        # JSON通信文件
        self.goal_file = 'control_goal.json'

        # 假设有一张从高空拍摄的地图/图片用于规划
        # 在实际仿真中，这可以是之前无人机飞过一圈拼出来的图，或者预置的卫星图
        self.map_image_path = "Disaster_economy_PPPs.jpg"
        self.meters_per_pixel = 0.12  # 根据 SLAM.py 中的参数

    def analyze_scene_and_plan(self):
        """
        调用感知模块(detector)和路径规划模块(pathfinding)
        生成真实的救援路径
        """
        if not os.path.exists(self.map_image_path):
            print(f"❌找不到地图图片 {self.map_image_path}")
            return []

        print(f"🔍正在分析灾区图像...")

        # 1. 读取图像 (逻辑来自 SLAM.py)
        img_bgr = cv2.imread(self.map_image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 2. 识别物体 (使用 detector.py)
        result = detect_objects(img_rgb)

        persons = []
        print(f" {len(result.boxes)} objects detected.")

        # 3. 提取幸存者坐标 (逻辑来自 SLAM.py)
        for box in result.boxes:
            name = result.names[int(box.cls)]
            mapped_class = map_yolo_class(name)

            if mapped_class == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 计算中心点并在物理坐标系中转换
                cx = (x1 + x2) / 2 * self.meters_per_pixel
                cy = (y1 + y2) / 2 * self.meters_per_pixel
                persons.append((cx, cy))
                print(f"find survivor at ({cx:.2f}meters, {cy:.2f}meters)")

        if not persons:
            print("No survivors were found. The mission is cancelled or the default hover is executed")
            return []

        # 4. 路径规划 (使用 pathfinding.py)
        # 假设无人机从 (0,0) 出发
        start_pos = (0, 0)
        print(f"🗺️Planning path: starting point {start_pos} -> {len(persons)} objects.")

        # 获取一系列密集的路径点 [(x,y), (x,y)...]
        raw_path = move_robot_toward_multiple_targets(start_pos, persons, step_size=0.5)

        # 5. 格式化为 teammate 控制器能读懂的 JSON 格式
        formatted_waypoints = []
        flight_height = 1.2  # 设定飞行高度

        # 为了减少通信量，可以每隔几个点取一个，或者直接全部发送
        # 这里将 pathfinding 生成的 2D 点转换为 3D 航点
        for p in raw_path:
            formatted_waypoints.append({
                "position": [float(p[0]), float(p[1]), flight_height],
                "altitude": flight_height
            })

        return formatted_waypoints

    def dispatch_mission(self, waypoints):
        if not waypoints:
            return

        mission_data = {
            "timestamp": time.time(),
            "mission_id": "RESCUE_PATH_V1",
            "waypoints": waypoints
        }

        try:
            with open(self.goal_file, 'w', encoding='utf-8') as f:
                json.dump(mission_data, f, indent=4)
            print(f"mission dispatched; contain {len(waypoints)} waypoints")
            print(f"press 'G' to start executing")
        except Exception as e:
            print(f"dispatch failed - {e}")


if __name__ == "__main__":
    commander = MissionCommander()

    # 执行分析与规划
    path = commander.analyze_scene_and_plan()

    # 下发任务
    commander.dispatch_mission(path)