from controller import Robot, Keyboard
import math
import os
import json
import numpy as np


class PID:
    #pid 控制器
    def __init__(self):
        # 初始化变量
        self.past_vx_error = 0.0
        self.past_vy_error = 0.0
        self.past_alt_error = 0.0
        self.past_pitch_error = 0.0
        self.past_roll_error = 0.0
        self.altitude_integrator = 0.0
        self.last_time = 0.0
        # edit: DM
        self.current_goal = None
        self.test_waypoints = [...]

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
        
        # ===== 测试轨迹=====
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

    def _read_goal(self):
        if not os.path.exists(self.goal_file):
            return None
        # edit: DM
        try:
            with open(self.goal_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('waypoints', [])
        except:
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
                self.waypoint_index = 0  # 重置路径点索引
                self.current_goal = None  # edit: DM 允许重新加载新任务
                self.state = 'GOTO'

            if self.state == 'MANUAL':
                step = 0.12
                if key == ord('W'):
                    self.manual_pitch_cmd = max(-0.3, self.manual_pitch_cmd - step)
                elif key == ord('S'):
                    self.manual_pitch_cmd = min(0.3, self.manual_pitch_cmd + step)
                
                elif key == ord('A'):
                    self.manual_yaw_rate_cmd = 0.5   # 左转
                elif key == ord('D'):
                    self.manual_yaw_rate_cmd = -0.5  # 右转
                
                if key == ord('Q'):
                    self.target_alt += self.manual_alt_step
                elif key == ord('E'):
                    self.target_alt = max(0.5, self.target_alt - self.manual_alt_step)
                
                # 偏航控制
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
        # 初始化电机
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
                # edit: DM
                if self.waypoint_index == 0 and not self.current_goal:
                    external_mission = self._read_goal()
                    if external_mission:
                        print("收到 Jiaqi 的任务指令，开始执行。")
                        self.test_waypoints = external_mission  # 覆盖原有路径
                        self.current_goal = True  # 标记已加载

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
                            else:
                                next_wp = self.test_waypoints[self.waypoint_index]
                        
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
                
            else:  
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


def main():
    ctrl = DroneController()
    ctrl.run()


if __name__ == '__main__':
    main()