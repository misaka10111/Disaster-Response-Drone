from controller import Robot, Keyboard
import math
import os
import json
import numpy as np
import cv2

# drone control.py
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

        # 传感器角度（相对机体 x 轴）
        self.sensor_infos = [
            ("front", self.range_front, 0.0),
            ("back", self.range_back, math.pi),
            ("left", self.range_left, math.pi / 2),
            ("right", self.range_right, -math.pi / 2),
        ]

        # 地图参数
        self.MAP_SIZE = map_size
        self.RESOLUTION = resolution
        self.MAP_ORIGIN = map_size // 2

        # 对外的离散占据图：0=未知, 1=free, 2=occupied
        self.occupancy = np.zeros((map_size, map_size), dtype=np.uint8)

        # 内部 log-odds 概率栅格，0≈50%
        self.log_odds = np.zeros((map_size, map_size), dtype=np.float32)
        self.L_FREE_UPDATE = -0.45   # 观测到 free 时的增量
        self.L_OCC_UPDATE = 0.9      # 观测到 occupied 时的增量
        self.L_MIN = -4.0
        self.L_MAX = 4.0
        self.L_FREE_THR = -0.4
        self.L_OCC_THR = 0.4

        self.trajectory = []

        # 建筑识别相关
        self.buildings = []
        self.building_detect_interval = 10
        self.MIN_BUILDING_AREA = 20

        self.step_count = 0

        # 位姿指数平滑，减少 GPS/IMU 抖动
        self.pose_inited = False
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.pose_alpha = 0.3  # 越大越“相信当前测量”

        print(f"[SLAM] 初始化完成：地图 {map_size}x{map_size}, 分辨率 {resolution} m/格")

    # ------------------ 坐标与 log-odds 工具 ------------------

    def world_to_grid(self, x, y):
        # 使用 round 而不是简单截断，减少系统偏差
        gx = int(round(x / self.RESOLUTION)) + self.MAP_ORIGIN
        gy = int(round(y / self.RESOLUTION)) + self.MAP_ORIGIN
        return gx, gy

    def grid_in_bounds(self, gx, gy):
        return 0 <= gx < self.MAP_SIZE and 0 <= gy < self.MAP_SIZE

    def _update_log_odds_cell(self, gx, gy, delta_l):
        """对单个格子更新 log-odds 并同步 occupancy"""
        if not self.grid_in_bounds(gx, gy):
            return
        l = self.log_odds[gy, gx] + delta_l
        l = max(self.L_MIN, min(self.L_MAX, l))
        self.log_odds[gy, gx] = l

        # 根据置信度阈值更新 0/1/2
        if l > self.L_OCC_THR:
            self.occupancy[gy, gx] = 2
        elif l < self.L_FREE_THR:
            self.occupancy[gy, gx] = 1
        else:
            self.occupancy[gy, gx] = 0

    # ------------------ Bresenham 射线算法 ------------------

    def _bresenham(self, x0, y0, x1, y1):
        """整数栅格上的 Bresenham 直线，返回 (gx,gy) 列表"""
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

    # ------------------ 单次测距融合 + 清理机器人附近 ------------------

    def _clear_robot_nearby(self):
        """每一步在机器人附近画一小圈 free，避免把自己当障碍"""
        gx0, gy0 = self.world_to_grid(self.x, self.y)
        radius_m = 0.1  # 物理半径
        r_cells = int(radius_m / self.RESOLUTION) + 1
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy <= r_cells * r_cells:
                    self._update_log_odds_cell(gx0 + dx, gy0 + dy,
                                               self.L_FREE_UPDATE * 0.5)

    def _integrate_range_measurement(self, x, y, yaw, sensor, rel_angle):
        """单个距离测量的 ray-casting（概率 + Bresenham）"""
        if sensor is None:
            return

        r_mm = sensor.getValue()
        max_r_mm = sensor.getMaxValue()

        if r_mm <= 1e-3 or max_r_mm <= 1e-3:
            return
        # 过小的值视为噪声
        if r_mm < 5.0:
            return

        hit_obstacle = True
        if r_mm >= 0.98 * max_r_mm:
            # 接近最大量程：没有击中障碍，只画 free
            hit_obstacle = False
            r_mm = max_r_mm

        r = r_mm / 1000.0  # mm->m
        angle = yaw + rel_angle

        # 起点、终点转换到栅格坐标
        gx0, gy0 = self.world_to_grid(x, y)
        wx_end = x + r * math.cos(angle)
        wy_end = y + r * math.sin(angle)
        gx1, gy1 = self.world_to_grid(wx_end, wy_end)

        # Bresenham 生成路径
        line_cells = self._bresenham(gx0, gy0, gx1, gy1)

        if len(line_cells) == 0:
            return

        # 路径中除了最后一个格子都视为 free
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

        # 如果真的有击中障碍，则将末端格子更新为 occupied
        if hit_obstacle:
            gx, gy = last_cell
            if self.grid_in_bounds(gx, gy):
                self._update_log_odds_cell(gx, gy, self.L_OCC_UPDATE)
        else:
            # 没击中障碍时，末端也看作 free（在可视范围内都是空的）
            gx, gy = last_cell
            if self.grid_in_bounds(gx, gy):
                self._update_log_odds_cell(gx, gy, self.L_FREE_UPDATE * 0.7)

    # ------------------ 建筑识别 + 显示 ------------------

    def _detect_buildings(self):
        # 使用形态学闭运算让障碍轮廓更完整
        binary = (self.occupancy == 2).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        binary_smooth = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_smooth, connectivity=8
        )
        buildings = []
        for i in range(1, num_labels):  # 0=背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.MIN_BUILDING_AREA:
                continue
            cx, cy = centroids[i]
            buildings.append((int(cx), int(cy), int(area)))
        self.buildings = buildings

    def _show_map(self):
        img = np.zeros((self.MAP_SIZE, self.MAP_SIZE, 3), dtype=np.uint8)

        # 未知：深灰；free：黑；occupied：白
        img[:, :] = (40, 40, 40)
        img[self.occupancy == 1] = (0, 0, 0)
        img[self.occupancy == 2] = (255, 255, 255)

        # 轨迹：绿色
        for (tx, ty) in self.trajectory:
            gx, gy = self.world_to_grid(tx, ty)
            if self.grid_in_bounds(gx, gy):
                img[gy, gx] = (0, 255, 0)

        # 建筑中心：蓝点 + B1/B2...
        for idx, (gx_b, gy_b, area) in enumerate(self.buildings, start=1):
            cv2.circle(img, (gx_b, gy_b), 2, (255, 0, 0), -1)
            cv2.putText(
                img, f"B{idx}", (gx_b + 2, gy_b - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1
            )

        # 当前无人机位置 + 朝向箭头
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

    # ------------------ 主更新 ------------------

    def update(self):
        self.step_count += 1

        if self.gps is None:
            return

        pos = self.gps.getValues()   # [x,y,z]
        meas_x = float(pos[0])
        meas_y = float(pos[1])

        if self.imu is not None:
            rpy = self.imu.getRollPitchYaw()
            meas_yaw = float(rpy[2])
        else:
            meas_yaw = 0.0

        # 位姿指数平滑
        if not self.pose_inited:
            self.x, self.y, self.yaw = meas_x, meas_y, meas_yaw
            self.pose_inited = True
        else:
            a = self.pose_alpha
            self.x = (1 - a) * self.x + a * meas_x
            self.y = (1 - a) * self.y + a * meas_y
            self.yaw = (1 - a) * self.yaw + a * meas_yaw

        # 记录轨迹（平滑后的）
        self.trajectory.append((self.x, self.y))

        # 清理机器人附近区域，避免“自己是障碍”
        self._clear_robot_nearby()

        # 对每个距离传感器做 ray-casting
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

        # 定期做建筑识别（使用平滑后的 binary）
        if self.step_count % self.building_detect_interval == 0:
            self._detect_buildings()

        # 显示地图
        self._show_map()

    def close(self):
        cv2.destroyAllWindows()


# 控制部分


class PID:
    # pid 控制器
    def __init__(self):
        # 初始化变量
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
        # PID参数
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

        # 计算俯仰和横滚误差
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
        # 初始化机器人
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

        # 新增：后向距离传感器（给 SLAM 用）
        self.range_back = None
        try:
            self.range_back = self.robot.getDevice('range_back')
            if self.range_back:
                self.range_back.enable(self.timestep)
        except Exception:
            self.range_back = None
            print("[SLAM] WARNING: 找不到 range_back 传感器，可选。")

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

        self.target_alt = 1.2  # 提高初始目标高度
        self.state = 'HOVER'
        self.goal_file = os.path.join(os.path.dirname(__file__), 'control_goal.json')
        self.current_goal = None

        # 测试轨迹
        self.test_waypoints = [
            {"position": [1.5, 0.0, 1.2], "altitude": 1.2},
            {"position": [2.0, 1.0, 1.5], "altitude": 1.5},
            {"position": [1.0, 2.0, 1.5], "altitude": 1.5},
            {"position": [0.0, 1.0, 1.2], "altitude": 1.2},
        ]
        self.waypoint_index = 0
        self.waypoint_threshold = 0.3  # 到达阈

        self.manual_roll_cmd = 0.0
        self.manual_pitch_cmd = 0.0
        self.manual_yaw_rate_cmd = 0.0
        self.manual_alt_step = 0.15  # 高度调整步长

        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_time = self.robot.getTime()

        self.takeoff_phase = True
        self.takeoff_duration = 3.0  # 起飞持续3秒

        # 初始化 SLAM
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
            print("[SLAM] 关键信号缺失（gps / imu / range_xx），SLAM 不启用。")

    def _read_goal(self):
        if not os.path.exists(self.goal_file):
            return None
        try:
            with open(self.goal_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"[Controller] 成功读取任务文件，包含 {len(data.get('waypoints', []))} 个航点")
                return data.get('waypoints', [])
        except Exception as e:
            print(f"[Controller] 读取任务文件失败: {e}")
            return None
        # with open(self.goal_file, 'r', encoding='utf-8') as f:
        #     return json.load(f)

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
        return (0.0, 0.0, 0.0)

    def _get_gyro(self):
        if self.gyro:
            gyro_vals = self.gyro.getValues()
            if gyro_vals:
                return gyro_vals
        return (0.0, 0.0, 0.0)

    def _get_body_velocity(self, dt):
        if not self.gps or dt <= 0:
            return 0.0, 0.0

        pos = self.gps.getValues()
        x, y = pos[0], pos[1]

        # 计算全局速度
        vx_global = (x - self.prev_x) / dt
        vy_global = (y - self.prev_y) / dt

        # 更新上一位置
        self.prev_x, self.prev_y = x, y

        _, _, yaw = self._get_rpy()
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        vx_body = vx_global * cos_yaw + vy_global * sin_yaw
        vy_body = -vx_global * sin_yaw + vy_global * cos_yaw

        return vx_body, vy_body

    def _handle_keyboard(self):
        # 每次循环前清零手动指令
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
                self.waypoint_index = 0  # reset waypoint index
                self.current_goal = None
                self.state = 'GOTO'
                print("[Controller] switch to GOTO mode, ready to receive instructions...")

            if self.state == 'MANUAL':
                step = 0.12
                step = 0.12
                if key == ord('W'):
                    self.manual_pitch_cmd = max(-0.3, self.manual_pitch_cmd - step)
                elif key == ord('S'):
                    self.manual_pitch_cmd = min(0.3, self.manual_pitch_cmd + step)

                # 这里可以改成横移而不是偏航，如果需要侧移的话
                elif key == ord('A'):
                    self.manual_yaw_rate_cmd = 0.5   # turn left
                elif key == ord('D'):
                    self.manual_yaw_rate_cmd = -0.5  # turn right

                if key == ord('Q'):
                    self.target_alt += self.manual_alt_step
                elif key == ord('E'):
                    self.target_alt = max(0.5, self.target_alt - self.manual_alt_step)

                # 额外偏航键
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

    def run(self):
        # 初始化电机：先给一个速度让它起飞
        if self.motors:
            for motor in self.motors:
                motor.setVelocity(20.0)
            for _ in range(5):
                self.robot.step(self.timestep)

        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime()
            dt = current_time - self.prev_time
            self.prev_time = current_time

            self._handle_keyboard()

            alt = self._get_altitude()
            roll, pitch, yaw = self._get_rpy()
            _, _, yaw_rate = self._get_gyro()
            vx_body, vy_body = self._get_body_velocity(dt)

            desired_altitude = self.target_alt  # 直接使用目标高度
            if self.state == 'MANUAL':
                k_vel = 0.6  # 速度映射系数
                desired_vx = -self.manual_pitch_cmd * k_vel
                desired_vy = self.manual_roll_cmd * k_vel
                desired_yaw_rate = self.manual_yaw_rate_cmd

            elif self.state == 'GOTO':
                if self.waypoint_index == 0 and not self.current_goal:
                    external_mission = self._read_goal()
                    if external_mission:
                        self.test_waypoints = external_mission
                        self.current_goal = True
                        print("[Controller] mission loaded, start executing")

                if self.waypoint_index < len(self.test_waypoints):
                    current_wp = self.test_waypoints[self.waypoint_index]
                    desired_altitude = float(current_wp.get('altitude', desired_altitude))
                    desired_yaw_rate = 0.0

                    if self.gps:
                        pos = self.gps.getValues()
                        px, py, pz = pos[0], pos[1], pos[2]
                        tx, ty, tz = current_wp['position']

                        ex = tx - px
                        ey = ty - py
                        distance = math.sqrt(ex**2 + ey**2)

                        if distance < self.waypoint_threshold:
                            self.waypoint_index += 1
                            if self.waypoint_index >= len(self.test_waypoints):
                                self.state = 'HOVER'
                                desired_vx = 0.0
                                desired_vy = 0.0

                        _, _, yaw = self._get_rpy()
                        cos_yaw = math.cos(yaw)
                        sin_yaw = math.sin(yaw)

                        ex_body = ex * cos_yaw + ey * sin_yaw
                        ey_body = -ex * sin_yaw + ey * cos_yaw

                        k_goto = 0.5  # 增益
                        desired_vx = k_goto * ex_body
                        desired_vy = k_goto * ey_body

                        max_vel = 0.5
                        desired_vx = max(-max_vel, min(max_vel, desired_vx))
                        desired_vy = max(-max_vel, min(max_vel, desired_vy))

                    else:
                        desired_vx = 0.0
                        desired_vy = 0.0
                else:
                    desired_vx = 0.0
                    desired_vy = 0.0

            else:  # HOVER
                desired_vx = 0.0
                desired_vy = 0.0
                desired_yaw_rate = 0.0

            if self.motors and len(self.motors) == 4:
                motor_cmds = self.low_level.pid(
                    dt,
                    desired_vx, desired_vy, desired_yaw_rate, desired_altitude,
                    roll, pitch, yaw_rate,
                    alt, vx_body, vy_body
                )

                self.motors[0].setVelocity(-motor_cmds[0])
                self.motors[1].setVelocity(motor_cmds[1])
                self.motors[2].setVelocity(-motor_cmds[2])
                self.motors[3].setVelocity(motor_cmds[3])

            # === SLAM 更新：每一步都画地图 ===
            if self.slam is not None:
                self.slam.update()


def main():
    ctrl = DroneController()
    ctrl.run()


if __name__ == '__main__':
    main()
