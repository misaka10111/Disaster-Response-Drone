# Disaster-Response-Drone

### Project Overview
This project implements an autonomous search and rescue drone simulation using Webots. It utilizes a Crazyflie drone with a 45° downward-facing camera to perform a "Lawnmower" search pattern, detect survivors using a custom YOLOv8 model, and autonomously generate rescue paths based on global metric coordinates.

---

## Implementation Details (Self-Implemented)
The following core logic and modules were designed and implemented specifically for this project:

### 1. Autonomous Flight Control (drone_control.py)
- **Finite State Machine (FSM)**: Implemented logic for switching between HOVER, MANUAL, SEARCHING, and GOTO states.
- **PID Controller Tuning**: Heavily tuned PID parameters for the Crazyflie. Implemented Error Clamping and Integrator Anti-windup to resolve instability during rapid descent and takeoff.
- **Search Strategy**: Developed a "Lawnmower" pattern generator and a "Stop-Scan-Go" strategy to stabilize the drone before capturing images.

### 2. Perception & Mission Planning (mission_commander.py)
- **YOLOv8 Logic**: Integrated the custom-trained YOLOv8 model for real-time inference.
- **Coordinate Transformation**:
	- **Dynamic Scale**: Calculates real-time meters_per_pixel based on altitude and FOV.
	- **Oblique Correction**: Mathematical correction for the 45-degree camera tilt to map pixels to accurate body-frame coordinates.
- **Spatial Clustering**: Implemented a de-duplication algorithm to merge close targets (<2.5m) and filter unreliable edge detections.
- **Memory System**: Created a JSON persistence layer to prevent double-counting survivors across different frames.

### 3. Data Pipeline
- Built an automated data collection mode to capture and save time-stamped images, used to create the training dataset.

## External Libraries & Pre-programmed Packages
- The following tools and code bases were used as foundations:
- **Simulation**: Webots R2023b & standard Crazyflie PROTO model.
- **Libraries**: `ultralytics` (YOLOv8), `opencv-python`, `numpy`, `controller` (Webots API).
- **Adapted Code**:
  - `SLAM.py`: Occupancy grid mapping and Lidar fusion logic (Adapted from course materials). Note: Used for visualization only; rescue navigation uses GPS.
  - `pathfinding.py`: Basic vector-stepping algorithm for generating dense waypoints (Used as a utility helper).

## Limitations & Known Issues
To ensure transparency regarding current system capabilities:
1. **Flight Stability**: Minor aerodynamic oscillations may occur during rapid descent. A 3-second stabilization buffer is required at each waypoint.
2. **Z-Axis Assumption**: Coordinate transformation assumes a flat ground (z=0) and does not account for survivors on varying terrain heights.
3. **Obstacle Avoidance**: The current pathfinding generates direct paths to targets and does not actively use the SLAM grid to avoid obstacles between waypoints.

## To Run to Simulation
1. Install dependencies
2. Open the world file in Webots
3. Assign the `drone_control` controller to the Crazyflie (Robot node)
4. Start simulation
5. Press 'G' to initiate the Autonomous Mission (Search -> Detect -> Rescue(GOTO))
