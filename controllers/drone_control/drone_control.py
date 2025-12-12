from controller import Robot, Keyboard
import math
import os
import json
import numpy as np
import cv2


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


# Proportional-Integral-Derivative Control
class PID:
    # pid controller
    def __init__(self):
        # Initialize variables
        self.past_vx_error = 0.0
        self.past_vy_error = 0.0
        self.past_alt_error = 0.0
        self.past_pitch_error = 0.0
        self.past_roll_error = 0.0
        self.altitude_integrator = 0.0
        self.last_time = 0.0

    def pid(self, dt, desired_vx, desired_vy, desired_yaw_rate, desired_altitude,
            actual_roll, actual_pitch, actual_yaw_rate,
            actual_altitude, actual_vx, actual_vy):
        # PID parameter
        gains = {
            "kp_att_y": 1, "kd_att_y": 0.5,
            "kp_att_rp": 0.5, "kd_att_rp": 0.1,
            "kp_vel_xy": 2, "kd_vel_xy": 0.5,
            "kp_z": 10, "ki_z": 5, "kd_z": 5
        }

        vx_error = desired_vx - actual_vx
        vx_deriv = (vx_error - self.past_vx_error) / dt
        vy_error = desired_vy - actual_vy
        vy_deriv = (vy_error - self.past_vy_error) / dt
        desired_pitch = gains["kp_vel_xy"] * np.clip(vx_error, -1, 1) + gains["kd_vel_xy"] * vx_deriv
        desired_roll = -gains["kp_vel_xy"] * np.clip(vy_error, -1, 1) - gains["kd_vel_xy"] * vy_deriv
        self.past_vx_error = vx_error
        self.past_vy_error = vy_error

        alt_error = desired_altitude - actual_altitude
        alt_deriv = (alt_error - self.past_alt_error) / dt
        self.altitude_integrator += alt_error * dt
        alt_command = gains["kp_z"] * alt_error + gains["kd_z"] * alt_deriv + \
            gains["ki_z"] * np.clip(self.altitude_integrator, -2, 2) + 48
        self.past_alt_error = alt_error

        # Calculate the pitch and roll errors
        pitch_error = desired_pitch - actual_pitch
        pitch_deriv = (pitch_error - self.past_pitch_error) / dt
        roll_error = desired_roll - actual_roll
        roll_deriv = (roll_error - self.past_roll_error) / dt
        yaw_rate_error = desired_yaw_rate - actual_yaw_rate
        roll_command = gains["kp_att_rp"] * np.clip(roll_error, -1, 1) + gains["kd_att_rp"] * roll_deriv
        pitch_command = -gains["kp_att_rp"] * np.clip(pitch_error, -1, 1) - gains["kd_att_rp"] * pitch_deriv
        yaw_command = gains["kp_att_y"] * np.clip(yaw_rate_error, -1, 1)
        self.past_pitch_error = pitch_error
        self.past_roll_error = roll_error

        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command

        m1 = np.clip(m1, 0, 600)
        m2 = np.clip(m2, 0, 600)
        m3 = np.clip(m3, 0, 600)
        m4 = np.clip(m4, 0, 600)

        return [m1, m2, m3, m4]


class DroneController:
    def __init__(self):
        # Initialize the robot
        self.has_scanned = False
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep()) or 32

        self.imu = self.robot.getDevice('inertial_unit')
        if self.imu:
            self.imu.enable(self.timestep)

        self.gyro = self.robot.getDevice('gyro')
        if self.gyro:
            self.gyro.enable(self.timestep)

        self.gps = self.robot.getDevice('gps')
        if self.gps:
            self.gps.enable(self.timestep)

        self.range_front = self.robot.getDevice('range_front')
        if self.range_front:
            self.range_front.enable(self.timestep)

        self.range_right = self.robot.getDevice('range_right')
        if self.range_right:
            self.range_right.enable(self.timestep)

        self.range_left = self.robot.getDevice('range_left')
        if self.range_left:
            self.range_left.enable(self.timestep)

        # Backward distance sensor (for SLAM)
        self.range_back = None
        try:
            self.range_back = self.robot.getDevice('range_back')
            if self.range_back:
                self.range_back.enable(self.timestep)
        except Exception:
            self.range_back = None
            print("[SLAM] WARNING: cannot find range_back sensor")

        self.camera = self.robot.getDevice('camera')
        if self.camera:
            self.camera.enable(self.timestep)

        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        self.motors = []
        motor_names = ['m1_motor', 'm2_motor', 'm3_motor', 'm4_motor']

        for name in motor_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)
                self.motors.append(motor)

        self.low_level = PID()

        self.target_alt = 1.2  # Increase the initial target height
        self.state = 'HOVER'
        self.goal_file = os.path.join(os.path.dirname(__file__), 'control_goal.json')
        self.current_goal = None

        # Test trajectory
        self.test_waypoints = [
            {"position": [1.5, 0.0, 1.2], "altitude": 1.2},
            {"position": [2.0, 1.0, 1.5], "altitude": 1.5},
            {"position": [1.0, 2.0, 1.5], "altitude": 1.5},
            {"position": [0.0, 1.0, 1.2], "altitude": 1.2},
        ]
        self.waypoint_index = 0
        self.waypoint_threshold = 0.3

        self.manual_roll_cmd = 0.0
        self.manual_pitch_cmd = 0.0
        self.manual_yaw_rate_cmd = 0.0
        self.manual_alt_step = 0.15  # Height adjustment step

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_time = self.robot.getTime()

        self.takeoff_phase = True
        self.takeoff_duration = 3.0  # 3 seconds

        # Initialize SLAM
        self.slam = None
        if self.gps and self.imu and self.range_front and self.range_left and self.range_right:
            self.slam = DroneSLAM(
                robot=self.robot,
                gps=self.gps,
                imu=self.imu,
                range_front=self.range_front,
                range_back=self.range_back,
                range_left=self.range_left,
                range_right=self.range_right,
                map_size=200,
                resolution=0.05
            )
        else:
            print("[SLAM] Missing key signals(gps/imu/range_xx), SLAM not started")

        # search mode variables
        self.search_waypoints = self._generate_search_pattern()
        self.search_index = 0
        self.is_scanning = False
        self.scan_timer = 0.0  # timer, used for hovering and taking photos

    # Take photo
    def _save_snapshot(self, save_to_dataset=False):
        if not self.camera:
            print("Camera not activated")
            return

        # 1. Obtain the original image data (BGRA form)
        raw_image = self.camera.getImage()
        if raw_image:
            width = self.camera.getWidth()
            height = self.camera.getHeight()

            # 2. Convert to numpy array
            img_array = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))

            # 3. Remove the Alpha channel and retain BGR (the default format of OpenCV)
            img_bgr = img_array[:, :, :3]

            # 4. Save file
            filename = "scan_result.jpg"
            cv2.imwrite(filename, img_bgr)
            print(f"Photo has save as {filename}")

            # Save photo for training
            if save_to_dataset:
                dataset_dir = "yolo_dataset"
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)

                timestamp = f"{self.robot.getTime():.2f}"
                filename = os.path.join(dataset_dir, f"img_{timestamp}.jpg")

                cv2.imwrite(filename, img_bgr)
                print(f"[Dataset] Image saved: {filename}")
            else:
                pass
        else:
            print("Cannot obtain image data")


    def _read_goal(self):
        if not os.path.exists(self.goal_file):
            return None
        try:
            with open(self.goal_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[Controller] The task file was successfully read, including {len(data.get('waypoints', []))} waypoints")
                return data.get('waypoints', [])
        except Exception as e:
            print(f"[Controller] Failed to read the task file: {e}")
            return None


    def _get_altitude(self):
        if self.gps:
            pos = self.gps.getValues()
            if len(pos) >= 3:
                return pos[2]
        return 0.0

    def _get_rpy(self):
        if self.imu:
            rpy = self.imu.getRollPitchYaw()
            if rpy:
                return rpy
        return 0.0, 0.0, 0.0

    def _get_gyro(self):
        if self.gyro:
            gyro_vals = self.gyro.getValues()
            if gyro_vals:
                return gyro_vals
        return 0.0, 0.0, 0.0

    def _get_body_velocity(self, dt):
        if not self.gps or dt <= 0:
            return 0.0, 0.0

        pos = self.gps.getValues()
        x, y = pos[0], pos[1]

        # Calculate the global velocity
        vx_global = (x - self.prev_x) / dt
        vy_global = (y - self.prev_y) / dt

        # Update the previous position
        self.prev_x, self.prev_y = x, y

        _, _, yaw = self._get_rpy()
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        vx_body = vx_global * cos_yaw + vy_global * sin_yaw
        vy_body = -vx_global * sin_yaw + vy_global * cos_yaw

        return vx_body, vy_body

    def _handle_keyboard(self):
        # Manual zeroing command before each cycle
        self.manual_pitch_cmd = 0.0
        self.manual_roll_cmd = 0.0
        self.manual_yaw_rate_cmd = 0.0

        key = self.keyboard.getKey()

        while key != -1:
            if key == ord('M') and self.state != 'MANUAL':
                self.state = 'MANUAL'

            elif key == ord('H') and self.state != 'HOVER':
                self.state = 'HOVER'
                self.target_alt = self._get_altitude()

            elif key == ord('G'):
                print("[Controller] Initial scan initiated...")

                # try to test it in place once firstly
                if self._run_mission_detection():
                    print("[Controller] Target found immediately!")
                    self.state = 'GOTO'
                    self.current_goal = None
                else:
                    print("[Controller] No target found. Starting SEARCH PATTERN.")
                    self.state = 'SEARCHING'
                    self.search_index = 0
                    self.is_scanning = False

            if self.state == 'MANUAL':
                step = 0.12
                if key == ord('W'):
                    self.manual_pitch_cmd = max(-0.3, self.manual_pitch_cmd - step)
                elif key == ord('S'):
                    self.manual_pitch_cmd = min(0.3, self.manual_pitch_cmd + step)

                elif key == ord('A'):
                    self.manual_yaw_rate_cmd = 0.5   # turn left
                elif key == ord('D'):
                    self.manual_yaw_rate_cmd = -0.5  # turn right

                if key == ord('Q'):
                    self.target_alt += self.manual_alt_step
                elif key == ord('E'):
                    self.target_alt = max(0.5, self.target_alt - self.manual_alt_step)

                if key == ord('R'):
                    self.manual_yaw_rate_cmd = 0.5
                elif key == ord('T'):
                    self.manual_yaw_rate_cmd = -0.5

            key = self.keyboard.getKey()

    def _takeoff_control(self):
        current_time = self.robot.getTime()
        if self.takeoff_phase and current_time < self.takeoff_duration:
            takeoff_alt = (current_time / self.takeoff_duration) * self.target_alt
            return takeoff_alt
        elif self.takeoff_phase:
            self.takeoff_phase = False
            return self.target_alt
        return self.target_alt

    def _generate_search_pattern(self):
        """
        Generate a serpentine path that covers the search area
        """
        waypoints = []
        x_start, x_end = -2.0, 2.0
        y_start, y_end = -2.0, 2.0
        step = 1.0  # search step
        altitude = 1.5

        y = y_start
        direction = 1  # 1 right, -1 left

        while y <= y_end:
            # Determine the starting and ending points of the journey based on the direction
            if direction == 1:
                row_points = np.arange(x_start, x_end + 0.1, step)
            else:
                row_points = np.arange(x_end, x_start - 0.1, -step)

            for x in row_points:
                waypoints.append({"position": [x, y, altitude], "altitude": altitude})

            y += step
            direction *= -1  # change direction

        print(f"[Search] Generated {len(waypoints)} search points.")
        return waypoints

    def _run_mission_detection(self):
        """
        Run mission_commander.py and check if there are any results
        Return: True (Found target), False (Not found)
        """
        # 1. delete old files
        if os.path.exists(self.goal_file):
            try:
                os.remove(self.goal_file)
            except:
                pass

        # 2. Call an external script
        import subprocess, sys
        try:
            subprocess.run([sys.executable, "mission_commander.py"], check=True)
        except Exception as e:
            print(f"Error running mission commander: {e}")
            return False

        # 3. read result
        mission_data = self._read_goal()
        if mission_data and len(mission_data) > 0:
            return True
        return False

    def run(self):
        # Initialize the motor
        if self.motors:
            for motor in self.motors:
                motor.setVelocity(20.0)
            for _ in range(5):
                self.robot.step(self.timestep)

        self.state = 'HOVER'
        self.target_alt = 1.5
        self.prev_time = self.robot.getTime()

        self.has_scanned = False

        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime()
            dt = current_time - self.prev_time
            self.prev_time = current_time

            self._handle_keyboard()

            alt = self._get_altitude()
            roll, pitch, yaw = self._get_rpy()
            _, _, yaw_rate = self._get_gyro()
            vx_body, vy_body = self._get_body_velocity(dt)

            if self.state == 'HOVER' and not self.has_scanned and current_time > 7.0:
                print(f"Hovering (height {alt:.2f}m), taking photo...")

                self._save_snapshot()
                self.has_scanned = True

                print("Press 'G' to start mission")

            desired_altitude = self.target_alt
            desired_vx = 0.0
            desired_vy = 0.0
            desired_yaw_rate = 0.0

            if self.state == 'MANUAL':
                k_vel = 0.6  # Velocity mapping coefficient
                desired_vx = -self.manual_pitch_cmd * k_vel
                desired_vy = self.manual_roll_cmd * k_vel
                desired_yaw_rate = self.manual_yaw_rate_cmd

            elif self.state == 'GOTO':
                if self.waypoint_index == 0 and not self.current_goal:
                    external_mission = self._read_goal()

                    if external_mission is not None:
                        self.current_goal = True

                        if len(external_mission) > 0:
                            self.test_waypoints = external_mission
                            print(f"Mission loaded: {len(external_mission)} waypoints")
                        else:
                            print("[Controller] Empty mission detected. Hovering.")
                            self.test_waypoints = []
                            self.state = 'HOVER'
                            continue

                # execute the waypoint flight logic
                if self.waypoint_index < len(self.test_waypoints):
                    current_wp = self.test_waypoints[self.waypoint_index]
                    desired_altitude = float(current_wp.get('altitude', self.target_alt))
                    desired_yaw_rate = 0.0

                    if self.gps:
                        pos = self.gps.getValues()
                        px, py = pos[0], pos[1]
                        tx, ty, _ = current_wp['position']

                        ex = tx - px
                        ey = ty - py
                        distance = math.sqrt(ex**2 + ey**2)

                        if distance < self.waypoint_threshold:
                            print(f"Arrive waypoint {self.waypoint_index + 1}/{len(self.test_waypoints)}")
                            self.waypoint_index += 1
                            if self.waypoint_index >= len(self.test_waypoints):
                                print("Mission accomplished. Hovering...")
                                self.state = 'HOVER'

                        _, _, yaw_curr = self._get_rpy()
                        cos_y = math.cos(yaw_curr)
                        sin_y = math.sin(yaw_curr)
                        ex_body = ex * cos_y + ey * sin_y
                        ey_body = -ex * sin_y + ey * cos_y

                        k_goto = 0.5  # speed coefficient
                        desired_vx = max(-0.5, min(0.5, k_goto * ex_body))
                        desired_vy = max(-0.5, min(0.5, k_goto * ey_body))
                else:
                    self.state = 'HOVER'

            elif self.state == 'SEARCHING':
                # 1. Check if the search is complete
                if self.search_index >= len(self.search_waypoints):
                    print("[Search] Area scanned completely. No targets found.")
                    self.state = 'HOVER'
                    self.target_alt = 1.75
                    continue

                # Get the search point you are currently going to
                target_wp = self.search_waypoints[self.search_index]
                desired_altitude = target_wp['altitude']

                # Calculate the distance
                pos = self.gps.getValues()
                dist = math.sqrt((target_wp['position'][0] - pos[0]) ** 2 +
                                 (target_wp['position'][1] - pos[1]) ** 2)

                # 2. Flight logic ( towards the search point)
                if dist > 0.2 and not self.is_scanning:
                    # not arrived
                    tx, ty, _ = target_wp['position']
                    ex = tx - pos[0]
                    ey = ty - pos[1]

                    # Machine coordinate conversion (for control)
                    _, _, yaw_curr = self._get_rpy()
                    cos_y = math.cos(yaw_curr)
                    sin_y = math.sin(yaw_curr)
                    desired_vx = 0.5 * (ex * cos_y + ey * sin_y)
                    desired_vy = 0.5 * (-ex * sin_y + ey * cos_y)

                    # Limit speed
                    desired_vx = max(-0.5, min(0.5, desired_vx))
                    desired_vy = max(-0.5, min(0.5, desired_vy))
                    desired_yaw_rate = 0.0

                # 3. Arrival (stop -> look -> go)
                else:
                    desired_vx = 0.0
                    desired_vy = 0.0

                    if not self.is_scanning:
                        print(f"[Search] Arrived at point {self.search_index}. Scanning...")
                        self.is_scanning = True
                        self.scan_timer = current_time

                    # take photo
                    if current_time - self.scan_timer > 2.0:
                        self._save_snapshot()

                        # call Mission Commander
                        print("[Search] Analyzing image...")
                        found_target = self._run_mission_detection()

                        if found_target:
                            print("[Search] !!! TARGET FOUND !!! Switching to GOTO.")
                            self.state = 'GOTO'
                            self.waypoint_index = 0
                            self.current_goal = None  # Trigger the read logic in GOTO
                            self.is_scanning = False  # reset flag
                        else:
                            print("[Search] Nothing here. Moving to next point.")
                            self.search_index += 1
                            self.is_scanning = False  # reset flag

            # PID output
            if self.motors and len(self.motors) == 4:
                motor_cmds = self.low_level.pid(
                    dt, desired_vx, desired_vy, desired_yaw_rate, desired_altitude,
                    roll, pitch, yaw_rate, alt, vx_body, vy_body
                )
                self.motors[0].setVelocity(-motor_cmds[0])
                self.motors[1].setVelocity(motor_cmds[1])
                self.motors[2].setVelocity(-motor_cmds[2])
                self.motors[3].setVelocity(motor_cmds[3])

            # draw map every step
            if self.slam is not None:
                self.slam.update()


def main():
    ctrl = DroneController()
    ctrl.run()


if __name__ == '__main__':
    main()
