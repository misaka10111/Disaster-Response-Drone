from controller import Robot, Keyboard
import math
import numpy as np
import cv2

# SLAM
class DroneSLAM:

    def __init__(self, robot, gps, imu,
                 range_front, range_back, range_left, range_right,
                 map_size=200, resolution=0.05):
        self.robot = robot
        self.gps = gps
        self.imu = imu
        self.range_front = range_front
        self.range_back = range_back
        self.range_left = range_left
        self.range_right = range_right

        self.time_step = int(robot.getBasicTimeStep())

        # Sensor angles (relative to body x-axis)
        self.sensor_infos = [
            ("front", self.range_front, 0.0),
            ("back", self.range_back, math.pi),
            ("left", self.range_left, math.pi / 2),
            ("right", self.range_right, -math.pi / 2),
        ]

        # Map parameters
        self.MAP_SIZE = map_size
        self.RESOLUTION = resolution
        self.MAP_ORIGIN = map_size // 2

        # Public discrete occupancy grid: 0=unknown, 1=free, 2=occupied
        self.occupancy = np.zeros((map_size, map_size), dtype=np.uint8)

        # Internal log-odds grid, 0 ≈ 50%
        self.log_odds = np.zeros((map_size, map_size), dtype=np.float32)
        self.L_FREE_UPDATE = -0.45   # Increment when cell is observed as free
        self.L_OCC_UPDATE = 0.9      # Increment when cell is observed as occupied
        self.L_MIN = -4.0
        self.L_MAX = 4.0
        self.L_FREE_THR = -0.4
        self.L_OCC_THR = 0.4

        self.trajectory = []

        # Building detection related
        self.buildings = []
        self.building_detect_interval = 10
        self.MIN_BUILDING_AREA = 20

        self.step_count = 0

        # Pose exponential smoothing to reduce GPS/IMU noise
        self.pose_inited = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.pose_alpha = 0.3  # Larger = trust current measurement more

        print(f"[SLAM] Initialized: map {map_size}x{map_size}, resolution {resolution} m/cell")

    #Coordinate & log-odds utilities

    def world_to_grid(self, x, y):
        # Use round instead of truncation to reduce systematic bias
        gx = int(round(x / self.RESOLUTION)) + self.MAP_ORIGIN
        gy = int(round(y / self.RESOLUTION)) + self.MAP_ORIGIN
        return gx, gy

    def grid_in_bounds(self, gx, gy):
        return 0 <= gx < self.MAP_SIZE and 0 <= gy < self.MAP_SIZE

    def _update_log_odds_cell(self, gx, gy, delta_l):
        #Update a single cell's log-odds and synchronize occupancy state.
        if not self.grid_in_bounds(gx, gy):
            return
        l = self.log_odds[gy, gx] + delta_l
        l = max(self.L_MIN, min(self.L_MAX, l))
        self.log_odds[gy, gx] = l

        # Update 0/1/2 label based on confidence thresholds
        if l > self.L_OCC_THR:
            self.occupancy[gy, gx] = 2
        elif l < self.L_FREE_THR:
            self.occupancy[gy, gx] = 1
        else:
            self.occupancy[gy, gx] = 0

    # Bresenham ray algorith

    def _bresenham(self, x0, y0, x1, y1):
       #Bresenham line on integer grid, returns list of (gx, gy).
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        gx, gy = x0, y0
        while True:
            points.append((gx, gy))
            if gx == x1 and gy == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                gx += sx
            if e2 < dx:
                err += dx
                gy += sy
        return points

    # Single range fusion + clearing near robot

    def _clear_robot_nearby(self):
        #Mark a small circle around the robot as free each step to avoid treating the robot think itself as an obstacle.
        gx0, gy0 = self.world_to_grid(self.x, self.y)
        radius_m = 0.1  # Physical radius
        r_cells = int(radius_m / self.RESOLUTION) + 1
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy <= r_cells * r_cells:
                    self._update_log_odds_cell(
                        gx0 + dx, gy0 + dy,
                        self.L_FREE_UPDATE * 0.5
                    )

    def _integrate_range_measurement(self, x, y, yaw, sensor, rel_angle):
        if sensor is None:
            return

        r_mm = sensor.getValue()
        max_r_mm = sensor.getMaxValue()

        if r_mm <= 1e-3 or max_r_mm <= 1e-3:
            return
        # Very small values are treated as noise
        if r_mm < 5.0:
            return

        hit_obstacle = True
        if r_mm >= 0.98 * max_r_mm:
            # Close to max range: assume no obstacle hit, only free space
            hit_obstacle = False
            r_mm = max_r_mm

        r = r_mm / 1000.0  # mm -> m
        angle = yaw + rel_angle

        # Convert start and end points to grid coordinates
        gx0, gy0 = self.world_to_grid(x, y)
        wx_end = x + r * math.cos(angle)
        wy_end = y + r * math.sin(angle)
        gx1, gy1 = self.world_to_grid(wx_end, wy_end)

        # Generate path using Bresenham
        line_cells = self._bresenham(gx0, gy0, gx1, gy1)

        if len(line_cells) == 0:
            return

        # All cells except the last are free cells
        if len(line_cells) >= 2:
            free_cells = line_cells[:-1]
            last_cell = line_cells[-1]
        else:
            free_cells = []
            last_cell = line_cells[0]

        for gx, gy in free_cells:
            if not self.grid_in_bounds(gx, gy):
                break
            self._update_log_odds_cell(gx, gy, self.L_FREE_UPDATE)

        # If we really hit an obstacle, update the end cell as occupied
        if hit_obstacle:
            gx, gy = last_cell
            if self.grid_in_bounds(gx, gy):
                self._update_log_odds_cell(gx, gy, self.L_OCC_UPDATE)
        else:
            # If no obstacle hit, treat the end cell as free as well
            gx, gy = last_cell
            if self.grid_in_bounds(gx, gy):
                self._update_log_odds_cell(gx, gy, self.L_FREE_UPDATE * 0.7)

    # Building detection + visualization

    def _detect_buildings(self):
        # Use morphological closing to make obstacle contours more complete
        binary = np.asarray(self.occupancy == 2).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        binary_smooth = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_smooth, connectivity=8
        )
        buildings = []
        for i in range(1, num_labels):  # 0 = background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.MIN_BUILDING_AREA:
                continue
            cx, cy = centroids[i]
            buildings.append((int(cx), int(cy), int(area)))
        self.buildings = buildings

    def _show_map(self):
        img = np.zeros((self.MAP_SIZE, self.MAP_SIZE, 3), dtype=np.uint8)

        # Unknown: dark gray; free: black; occupied: white
        img[:, :] = (40, 40, 40)
        img[self.occupancy == 1] = (0, 0, 0)
        img[self.occupancy == 2] = (255, 255, 255)

        # Trajectory: green
        for (tx, ty) in self.trajectory:
            gx, gy = self.world_to_grid(tx, ty)
            if self.grid_in_bounds(gx, gy):
                img[gy, gx] = (0, 255, 0)

        # Building centers: blue dots + B1/B2/
        for idx, (gx_b, gy_b, area) in enumerate(self.buildings, start=1):
            cv2.circle(img, (gx_b, gy_b), 2, (255, 0, 0), -1)
            cv2.putText(
                img, f"B{idx}", (gx_b + 2, gy_b - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1
            )

        # Current drone position + heading arrow
        gx_r, gy_r = self.world_to_grid(self.x, self.y)
        if self.grid_in_bounds(gx_r, gy_r):
            cv2.circle(img, (gx_r, gy_r), 3, (0, 0, 255), -1)
            arrow_len = 8
            ex = int(gx_r + arrow_len * math.cos(self.yaw))
            ey = int(gy_r + arrow_len * math.sin(self.yaw))
            cv2.arrowedLine(
                img, (gx_r, gy_r), (ex, ey),
                (0, 0, 255), 1, tipLength=0.4
            )

        img_big = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Drone SLAM Map", img_big)
        cv2.waitKey(1)

    # Main update

    def update(self):
        self.step_count += 1

        if self.gps is None:
            return

        pos = self.gps.getValues()   # [x, y, z]
        meas_x = float(pos[0])
        meas_y = float(pos[1])

        if self.imu is not None:
            rpy = self.imu.getRollPitchYaw()
            meas_yaw = float(rpy[2])
        else:
            meas_yaw = 0.0

        # Pose exponential smoothing
        if not self.pose_inited:
            self.x, self.y, self.yaw = meas_x, meas_y, meas_yaw
            self.pose_inited = True
        else:
            a = self.pose_alpha
            self.x = (1 - a) * self.x + a * meas_x
            self.y = (1 - a) * self.y + a * meas_y
            self.yaw = (1 - a) * self.yaw + a * meas_yaw

        # Record trajectory (smoothed pose)
        self.trajectory.append((self.x, self.y))

        # Clear area around the robot to avoid treating it as an obstacle
        self._clear_robot_nearby()

        # Perform ray-casting for each range sensor
        self._integrate_range_measurement(
            self.x, self.y, self.yaw, self.range_front, 0.0
        )
        self._integrate_range_measurement(
            self.x, self.y, self.yaw, self.range_back, math.pi
        )
        self._integrate_range_measurement(
            self.x, self.y, self.yaw, self.range_left, math.pi / 2
        )
        self._integrate_range_measurement(
            self.x, self.y, self.yaw, self.range_right, -math.pi / 2
        )

        # Periodically run building detection (on smoothed binary map)
        if self.step_count % self.building_detect_interval == 0:
            self._detect_buildings()

        # Display map
        self._show_map()

    def close(self):
        cv2.destroyAllWindows()

