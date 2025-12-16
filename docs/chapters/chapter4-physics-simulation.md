---
title: "Chapter 4: Physics Simulation and Digital Twins"
sidebar_position: 4
description: Introduction to digital twins, Gazebo physics simulation, gravity, collisions, and joint dynamics for humanoid robotics development.
---

# Chapter 4: Physics Simulation and Digital Twins

## Module 2: Simulation Foundations

## Learning Objectives

- Define the concept of a digital twin and its importance in humanoid robot development
- Understand Gazebo as a physics simulation platform and its integration with ROS 2
- Explain how gravity, collisions, and friction are modeled in robotics simulators
- Analyze joint dynamics including torque, velocity limits, and damping characteristics
- Apply physics simulation concepts to validate humanoid robot designs before hardware deployment

---

## Part 1: Introduction to Digital Twins and Physics Simulation

### What is a Digital Twin?

A **Digital Twin** is a virtual replica of a physical system that mirrors its real-world counterpart in real-time or near-real-time. In robotics, a digital twin encompasses:

- **Geometric fidelity**: Accurate 3D representation of the robot's physical structure
- **Kinematic accuracy**: Precise joint configurations and motion constraints
- **Dynamic properties**: Mass, inertia, friction, and material characteristics
- **Sensor simulation**: Virtual sensors producing data streams matching real hardware
- **Control interfaces**: Identical APIs for commanding the virtual and physical robot

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Digital Twin Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Physical World                              Virtual World                  │
│   ──────────────                              ─────────────                  │
│                                                                              │
│   ┌─────────────────┐                        ┌─────────────────┐            │
│   │   Physical      │◄──── Synchronization ──►│   Digital       │            │
│   │   Humanoid      │                        │   Twin          │            │
│   │   Robot         │                        │   (Gazebo)      │            │
│   └────────┬────────┘                        └────────┬────────┘            │
│            │                                          │                      │
│            ▼                                          ▼                      │
│   ┌─────────────────┐                        ┌─────────────────┐            │
│   │   Real Sensors  │                        │ Simulated       │            │
│   │   - Cameras     │                        │ Sensors         │            │
│   │   - IMU         │                        │ - Virtual Cam   │            │
│   │   - Force/Torque│                        │ - IMU Plugin    │            │
│   └────────┬────────┘                        └────────┬────────┘            │
│            │                                          │                      │
│            └──────────────┬───────────────────────────┘                      │
│                           ▼                                                  │
│                  ┌─────────────────┐                                        │
│                  │   ROS 2         │                                        │
│                  │   Control Stack │                                        │
│                  │   (Same Code)   │                                        │
│                  └─────────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benefits of Digital Twins in Humanoid Robotics

| Benefit | Description | Humanoid Application |
|---------|-------------|---------------------|
| **Risk-Free Testing** | Test dangerous maneuvers without hardware damage | Fall recovery, extreme poses |
| **Rapid Iteration** | Modify and test designs in minutes vs. weeks | Joint configuration tuning |
| **Parallel Development** | Software teams work without physical robot access | Control algorithm development |
| **Regression Testing** | Automated testing of control software | Continuous integration pipelines |
| **Training Data** | Generate scenarios for machine learning | Reinforcement learning for locomotion |
| **Failure Analysis** | Reproduce and analyze failure modes | Debug balance control issues |

### The Simulation-Reality Gap

While digital twins provide immense value, they inherently differ from physical systems:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Simulation vs Reality Differences                    │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Simulation                           Reality                          │
│   ──────────                           ───────                          │
│                                                                         │
│   • Perfect sensor data                • Noisy, delayed measurements   │
│   • Idealized physics models           • Complex material interactions │
│   • Deterministic execution            • Stochastic disturbances       │
│   • Uniform environments               • Varied, unpredictable terrain │
│   • Instant reset capability           • Time-consuming recovery       │
│   • Simplified contact models          • Complex deformation physics   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

Bridging this gap requires:
- **Domain randomization**: Varying simulation parameters during training
- **System identification**: Calibrating simulation to match real hardware
- **Robust control**: Designing controllers tolerant of model uncertainty

---

## Introduction to Gazebo

### What is Gazebo?

**Gazebo** is an open-source 3D robotics simulator that provides accurate physics simulation, sensor modeling, and environment rendering. Originally developed at Willow Garage alongside ROS, Gazebo has become the standard simulation platform for robotics research and development.

### Gazebo Versions

| Version | Name | Physics Engine | ROS Integration | Status |
|---------|------|---------------|-----------------|--------|
| Gazebo Classic | Gazebo 11 | ODE, Bullet, DART, Simbody | ROS 1, ROS 2 | Maintenance |
| Gazebo Sim | Ignition/Gz | DART, Bullet, TPE | ROS 2 native | Active Development |

For modern humanoid development with ROS 2, **Gazebo Sim** (formerly Ignition Gazebo) is recommended.

### Gazebo Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Gazebo Sim Architecture                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Gazebo Server (gzserver)                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │   World     │  │   Physics   │  │   Sensors   │  │   Plugins   │  │   │
│  │  │   Manager   │  │   Engine    │  │   Manager   │  │   System    │  │   │
│  │  │             │  │  (DART/     │  │             │  │             │  │   │
│  │  │  - Models   │  │   Bullet)   │  │  - Camera   │  │  - ROS 2    │  │   │
│  │  │  - Lights   │  │             │  │  - LiDAR    │  │  - Control  │  │   │
│  │  │  - Physics  │  │  - Gravity  │  │  - IMU      │  │  - Custom   │  │   │
│  │  │             │  │  - Contacts │  │  - F/T      │  │             │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      │                                       │
│                                      ▼                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Gazebo Client (gzclient)                      │   │
│  │                                                                        │   │
│  │   ┌────────────────┐    ┌────────────────┐    ┌────────────────┐     │   │
│  │   │   3D Renderer  │    │   GUI Panels   │    │   Visualization │     │   │
│  │   │   (OGRE2)      │    │   & Tools      │    │   Plugins       │     │   │
│  │   └────────────────┘    └────────────────┘    └────────────────┘     │   │
│  │                                                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Gazebo-ROS 2 Integration

Gazebo integrates with ROS 2 through the `ros_gz` bridge, enabling:

- Publishing simulated sensor data to ROS 2 topics
- Subscribing to ROS 2 commands for robot control
- Using standard ROS 2 tools (RViz, tf2) with simulated robots

```python
# Example: Launching Gazebo with ROS 2 integration
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to Gazebo launch file
    gz_sim_share = get_package_share_directory('ros_gz_sim')

    # Launch Gazebo with empty world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_sim_share, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items()
    )

    # Spawn humanoid robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'humanoid',
            '-file', '/path/to/humanoid.urdf',
            '-x', '0', '-y', '0', '-z', '1.0'
        ],
        output='screen'
    )

    # Bridge for ROS 2 communication
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
        ],
        output='screen'
    )

    return LaunchDescription([gazebo, spawn_robot, bridge])
```

---

## Physics Simulation Fundamentals

### Why Physics Accuracy Matters for Humanoids

Humanoid robots operate at the boundary of stability. Unlike wheeled robots with inherent static stability, humanoids must actively maintain balance through continuous control. This makes accurate physics simulation critical:

- **Balance control** depends on precise force and torque calculations
- **Locomotion** requires accurate ground contact modeling
- **Manipulation** needs realistic friction and collision responses
- **Fall prediction** relies on accurate dynamic simulation

---

## Gravity Simulation

### Modeling Gravitational Forces

Gravity is the most fundamental force affecting humanoid robots. In simulation, gravity applies a constant acceleration to all bodies:

$$F_g = m \cdot g$$

Where:
- $F_g$ = Gravitational force (N)
- $m$ = Mass of the body (kg)
- $g$ = Gravitational acceleration vector (typically [0, 0, -9.81] m/s²)

### Configuring Gravity in Gazebo

```xml
<!-- SDF world file with gravity configuration -->
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="humanoid_world">

    <!-- Standard Earth gravity -->
    <gravity>0 0 -9.81</gravity>

    <!-- Physics engine configuration -->
    <physics name="default_physics" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>

    <!-- Ground plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane>
              <normal>0 0 1</normal>
              <size>100 100</size>
            </plane>
          </geometry>
        </collision>
      </link>
    </model>

  </world>
</sdf>
```

### Gravity's Impact on Humanoid Simulation

| Aspect | Effect | Simulation Consideration |
|--------|--------|-------------------------|
| **Center of Mass** | Determines stability region | Must be accurately modeled per link |
| **Joint Torques** | Gravity compensation required | Affects motor sizing and control |
| **Ground Reaction Forces** | Balances gravitational load | Contact model accuracy critical |
| **Energy Consumption** | Continuous work against gravity | Battery life estimation |
| **Fall Dynamics** | Acceleration during falls | Safety system testing |

### Gravity Compensation in Control

Humanoid controllers must continuously compensate for gravity:

```python
# Gravity compensation example for humanoid joints
import numpy as np

class GravityCompensation:
    """
    Computes gravity compensation torques for humanoid robot.
    """

    def __init__(self, robot_model):
        self.model = robot_model
        self.g = np.array([0, 0, -9.81])

    def compute_torques(self, joint_positions):
        """
        Calculate joint torques needed to counteract gravity.

        Args:
            joint_positions: Current joint angle configuration

        Returns:
            numpy.ndarray: Gravity compensation torques per joint
        """
        # Compute mass matrix and gravity vector using robot dynamics
        # M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = tau

        # For gravity compensation, we need G(q)
        gravity_torques = self.model.compute_gravity_vector(
            joint_positions,
            self.g
        )

        return gravity_torques

    def feedforward_control(self, joint_positions, desired_torques):
        """
        Add gravity compensation to desired control torques.
        """
        gravity_comp = self.compute_torques(joint_positions)
        return desired_torques + gravity_comp
```

---

## Collision Simulation

### The Role of Collisions in Robotics

Collision detection and response are essential for:

- **Ground contact**: Enabling walking and standing
- **Object manipulation**: Grasping and carrying objects
- **Safety**: Detecting and responding to impacts
- **Environment interaction**: Navigating through spaces

### Collision Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Collision Detection Pipeline                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Broad Phase                    2. Narrow Phase                       │
│  ──────────────                    ───────────────                       │
│                                                                          │
│  ┌─────────────────┐              ┌─────────────────┐                   │
│  │  Spatial        │              │  Exact Geometry │                   │
│  │  Partitioning   │─────────────▶│  Intersection   │                   │
│  │                 │              │                 │                   │
│  │  - AABB Trees   │   Candidate  │  - GJK/EPA      │                   │
│  │  - Octrees      │     Pairs    │  - SAT          │                   │
│  │  - Sweep/Prune  │              │  - Mesh-Mesh    │                   │
│  └─────────────────┘              └────────┬────────┘                   │
│                                            │                             │
│                                            ▼                             │
│                               ┌─────────────────────┐                   │
│  3. Contact Generation        │   Contact Points    │                   │
│  ─────────────────────        │   - Position        │                   │
│                               │   - Normal          │                   │
│                               │   - Penetration     │                   │
│                               └────────┬────────────┘                   │
│                                        │                                 │
│                                        ▼                                 │
│                               ┌─────────────────────┐                   │
│  4. Contact Response          │   Force/Impulse     │                   │
│  ───────────────────          │   Calculation       │                   │
│                               │   - Normal force    │                   │
│                               │   - Friction        │                   │
│                               └─────────────────────┘                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Collision Geometry Types

| Geometry | Computation Cost | Accuracy | Use Case |
|----------|-----------------|----------|----------|
| **Sphere** | Very Low | Low | Quick approximations |
| **Box** | Low | Medium | Torso, feet bounding |
| **Cylinder** | Low | Medium | Limb segments |
| **Capsule** | Low | Good | Arms, legs |
| **Convex Mesh** | Medium | High | Complex link shapes |
| **Triangle Mesh** | High | Exact | Detailed collision |

### Configuring Collisions in URDF/SDF

```xml
<!-- URDF collision configuration for humanoid foot -->
<link name="left_foot">
  <!-- Visual geometry (detailed mesh) -->
  <visual>
    <geometry>
      <mesh filename="package://humanoid/meshes/left_foot.dae"/>
    </geometry>
  </visual>

  <!-- Collision geometry (simplified for performance) -->
  <collision>
    <origin xyz="0.05 0 -0.02" rpy="0 0 0"/>
    <geometry>
      <box size="0.20 0.10 0.04"/>
    </geometry>
  </collision>

  <!-- Inertial properties -->
  <inertial>
    <mass value="1.2"/>
    <inertia ixx="0.002" ixy="0" ixz="0"
             iyy="0.004" iyz="0" izz="0.003"/>
  </inertial>
</link>
```

### Friction Models

Friction is critical for humanoid locomotion—without friction, the robot cannot walk:

**Coulomb Friction Model:**

$$F_f \leq \mu \cdot F_n$$

Where:
- $F_f$ = Friction force (tangential)
- $\mu$ = Coefficient of friction
- $F_n$ = Normal force

```xml
<!-- SDF surface properties with friction -->
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>      <!-- Static friction coefficient -->
      <mu2>0.8</mu2>    <!-- Dynamic friction coefficient -->
      <fdir1>1 0 0</fdir1>  <!-- Primary friction direction -->
    </ode>
  </friction>
  <contact>
    <ode>
      <kp>1e6</kp>      <!-- Contact stiffness -->
      <kd>100</kd>      <!-- Contact damping -->
      <max_vel>0.1</max_vel>
      <min_depth>0.001</min_depth>
    </ode>
  </contact>
</surface>
```

### Ground Contact for Bipedal Walking

Accurate foot-ground contact is essential for humanoid locomotion:

```python
# Ground contact monitoring for humanoid
from rclpy.node import Node
from gazebo_msgs.msg import ContactsState

class FootContactMonitor(Node):
    """
    Monitors foot-ground contact for gait control.
    """

    def __init__(self):
        super().__init__('foot_contact_monitor')

        # Subscribe to contact sensor topics
        self.left_foot_sub = self.create_subscription(
            ContactsState,
            '/left_foot/contact',
            self.left_foot_callback,
            10
        )

        self.right_foot_sub = self.create_subscription(
            ContactsState,
            '/right_foot/contact',
            self.right_foot_callback,
            10
        )

        self.left_contact = False
        self.right_contact = False

    def left_foot_callback(self, msg: ContactsState):
        """Process left foot contact state."""
        self.left_contact = len(msg.states) > 0

        if self.left_contact:
            # Extract contact information
            contact = msg.states[0]
            self.get_logger().debug(
                f'Left foot contact: '
                f'force={contact.total_wrench.force.z:.2f}N'
            )

    def right_foot_callback(self, msg: ContactsState):
        """Process right foot contact state."""
        self.right_contact = len(msg.states) > 0

    def get_stance_phase(self):
        """
        Determine current stance phase.

        Returns:
            str: 'double_support', 'left_stance', 'right_stance', or 'flight'
        """
        if self.left_contact and self.right_contact:
            return 'double_support'
        elif self.left_contact:
            return 'left_stance'
        elif self.right_contact:
            return 'right_stance'
        else:
            return 'flight'
```

---

## Joint Dynamics

### Understanding Robot Joints

Joints connect rigid links and define their relative motion. For humanoids, joints enable the degrees of freedom necessary for locomotion and manipulation.

### Joint Types in Humanoid Robots

| Joint Type | DOF | Motion | Humanoid Application |
|------------|-----|--------|---------------------|
| **Revolute** | 1 | Rotation about axis | Knee, elbow, ankle pitch |
| **Continuous** | 1 | Unlimited rotation | Wheel joints (not typical) |
| **Prismatic** | 1 | Linear translation | Telescoping mechanisms |
| **Fixed** | 0 | No motion | Sensor mounts |
| **Floating** | 6 | Free motion | Base link (implicit) |
| **Ball (Spherical)** | 3 | 3-axis rotation | Hip, shoulder (approximated) |

### Joint Dynamics Model

The dynamics of a robot joint include:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Joint Dynamics Model                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Applied     ┌─────────────────────────────────────┐     Resulting     │
│   Torque  ───▶│                                     │───▶ Motion        │
│               │   τ = I·α + b·ω + τ_friction + τ_g  │                   │
│               │                                     │                   │
│               │   Where:                            │                   │
│               │   τ = Applied torque                │                   │
│               │   I = Moment of inertia             │                   │
│               │   α = Angular acceleration          │                   │
│               │   b = Viscous damping               │                   │
│               │   ω = Angular velocity              │                   │
│               │   τ_friction = Coulomb friction     │                   │
│               │   τ_g = Gravity torque              │                   │
│               │                                     │                   │
│               └─────────────────────────────────────┘                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuring Joint Dynamics in URDF

```xml
<!-- URDF joint configuration with dynamics -->
<joint name="left_knee" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>

  <!-- Joint axis of rotation -->
  <axis xyz="0 1 0"/>

  <!-- Joint origin relative to parent -->
  <origin xyz="0 0 -0.4" rpy="0 0 0"/>

  <!-- Motion limits -->
  <limit
    lower="0.0"           <!-- Minimum angle (rad) -->
    upper="2.5"           <!-- Maximum angle (rad) -->
    effort="200.0"        <!-- Maximum torque (Nm) -->
    velocity="5.0"        <!-- Maximum velocity (rad/s) -->
  />

  <!-- Dynamic properties -->
  <dynamics
    damping="0.5"         <!-- Viscous damping (Nm·s/rad) -->
    friction="0.1"        <!-- Coulomb friction (Nm) -->
  />
</joint>
```

### Joint Limits and Their Importance

| Limit Type | Purpose | Consequence of Violation |
|------------|---------|-------------------------|
| **Position** | Prevent overextension | Mechanical damage, singularities |
| **Velocity** | Motor speed constraints | Motor burnout, loss of control |
| **Effort** | Torque/force limits | Actuator saturation, instability |
| **Acceleration** | Smooth motion | Jerky movements, vibration |

### Joint Control Modes

```python
# Joint control modes for humanoid simulation
from enum import Enum
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray

class JointControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    EFFORT = "effort"

class HumanoidJointController(Node):
    """
    Multi-mode joint controller for humanoid robot.
    """

    def __init__(self):
        super().__init__('humanoid_joint_controller')

        # Position control publisher
        self.position_pub = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Effort control publisher
        self.effort_pub = self.create_publisher(
            Float64MultiArray,
            '/effort_controller/commands',
            10
        )

        self.joint_names = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee', 'right_ankle_pitch', 'right_ankle_roll'
        ]

    def send_position_command(self, positions: list, duration: float = 1.0):
        """
        Send position command to joints.

        Args:
            positions: Target joint positions (rad)
            duration: Time to reach target (s)
        """
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)

        trajectory.points = [point]
        self.position_pub.publish(trajectory)

    def send_effort_command(self, torques: list):
        """
        Send effort (torque) command to joints.

        Args:
            torques: Target joint torques (Nm)
        """
        msg = Float64MultiArray()
        msg.data = torques
        self.effort_pub.publish(msg)

    def compute_inverse_dynamics(self, positions, velocities, accelerations):
        """
        Compute required torques for desired motion.

        Uses robot dynamics: τ = M(q)·q̈ + C(q,q̇)·q̇ + G(q)

        Returns:
            list: Required joint torques
        """
        # Implementation would use robot dynamics library
        # (e.g., pinocchio, rbdl, or custom implementation)
        pass
```

### Simulating Joint Dynamics Accurately

Key considerations for realistic joint simulation:

1. **Inertia Matching**: Ensure simulated link inertias match physical robot
2. **Friction Modeling**: Include both static and viscous friction
3. **Backlash**: Model gear backlash for accurate position control
4. **Flexibility**: Consider joint/link flexibility for high-precision tasks
5. **Actuator Dynamics**: Include motor dynamics (not just ideal torque sources)

```python
# Joint dynamics parameters for realistic simulation
joint_dynamics_params = {
    'left_knee': {
        'rotor_inertia': 0.01,      # kg·m² (motor rotor)
        'gear_ratio': 100,           # Reduction ratio
        'gear_efficiency': 0.85,     # Power transmission efficiency
        'viscous_damping': 0.5,      # Nm·s/rad
        'coulomb_friction': 0.1,     # Nm
        'backlash': 0.001,           # rad
        'torque_constant': 0.1,      # Nm/A
        'max_current': 20.0,         # A
    },
    # ... other joints
}
```

---

## Part 2: SDF and World Configuration

### Understanding SDF (Simulation Description Format)

**SDF (Simulation Description Format)** is an XML-based format designed specifically for describing simulation environments, robots, and sensors. While URDF was created for robot description in ROS, SDF was developed for Gazebo to provide more comprehensive simulation capabilities.

### SDF vs URDF Comparison

| Feature | URDF | SDF |
|---------|------|-----|
| **Primary Purpose** | Robot description | Complete simulation worlds |
| **World Definition** | Not supported | Full support (terrain, lighting, physics) |
| **Multiple Robots** | Single robot per file | Multiple models in one world |
| **Sensor Plugins** | Limited (via Gazebo extensions) | Native support |
| **Physics Properties** | Basic | Comprehensive |
| **Lighting** | Not supported | Full lighting system |
| **Nested Models** | Not supported | Supported |
| **File Extension** | `.urdf`, `.xacro` | `.sdf`, `.world` |

### SDF Structure Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SDF File Hierarchy                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  <sdf>                                                                   │
│  └── <world>                        # Simulation environment             │
│      ├── <physics>                  # Physics engine settings            │
│      ├── <gravity>                  # Gravity vector                     │
│      ├── <scene>                    # Visual settings                    │
│      ├── <light>                    # Light sources                      │
│      ├── <model>                    # Robot or object                    │
│      │   ├── <link>                 # Rigid body                         │
│      │   │   ├── <visual>           # Rendering geometry                 │
│      │   │   ├── <collision>        # Physics geometry                   │
│      │   │   ├── <inertial>         # Mass properties                    │
│      │   │   └── <sensor>           # Attached sensors                   │
│      │   ├── <joint>                # Link connections                   │
│      │   └── <plugin>               # Custom behaviors                   │
│      └── <plugin>                   # World-level plugins                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### URDF to SDF Conversion

When you spawn a URDF robot in Gazebo, it automatically converts to SDF internally. However, you can also manually convert:

```bash
# Convert URDF to SDF using gz tool
gz sdf -p robot.urdf > robot.sdf
```

Key differences during conversion:
- URDF `<gazebo>` tags become native SDF elements
- Material colors map to SDF `<material>` elements
- Gazebo plugins transfer to SDF `<plugin>` elements

---

### Gazebo World Configuration

A Gazebo world file defines the complete simulation environment. Here's a concise example demonstrating gravity and lighting configuration:

```xml
<?xml version="1.0" ?>
<sdf version="1.8">
  <world name="humanoid_training_world">

    <!-- Gravity Configuration -->
    <!-- Standard Earth gravity pointing down (-Z) -->
    <gravity>0 0 -9.81</gravity>

    <!-- Physics Engine Settings -->
    <physics name="default_physics" type="dart">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <!-- Sunlight (Directional Light) -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>

    <!-- Ambient Light for Overall Illumination -->
    <light name="ambient" type="point">
      <pose>0 0 5 0 0 0</pose>
      <diffuse>0.4 0.4 0.4 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <attenuation>
        <range>50</range>
        <constant>0.5</constant>
        <linear>0.01</linear>
      </attenuation>
    </light>

    <!-- Ground Plane -->
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <plane><normal>0 0 1</normal></plane>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <plane><normal>0 0 1</normal><size>50 50</size></plane>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
          </material>
        </visual>
      </link>
    </model>

  </world>
</sdf>
```

### Understanding Key World Elements

#### Gravity Vector

The `<gravity>` element defines the gravitational acceleration as a 3D vector (x, y, z):

| Configuration | Vector | Use Case |
|---------------|--------|----------|
| Earth gravity | `0 0 -9.81` | Standard humanoid simulation |
| Moon gravity | `0 0 -1.62` | Low-gravity locomotion testing |
| Zero gravity | `0 0 0` | Space robotics simulation |
| Tilted gravity | `1 0 -9.81` | Slope/incline simulation |

#### Light Types

| Light Type | Description | Best For |
|------------|-------------|----------|
| **Directional** | Parallel rays (sun-like) | Outdoor scenes, shadows |
| **Point** | Radiates in all directions | Indoor ambient lighting |
| **Spot** | Cone-shaped beam | Focused illumination |

---

### Launching Gazebo Worlds with ROS 2

To launch a Gazebo simulation with ROS 2 integration, use the `ros_gz_sim` package:

```bash
# Launch Gazebo Sim with an empty world
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:="-r empty.sdf"

# Launch with a custom world file
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:="-r /path/to/humanoid_world.sdf"

# Launch with verbose output for debugging
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:="-v 4 -r humanoid_world.sdf"
```

### Complete ROS 2 Launch File for Humanoid Simulation

```python
# humanoid_gazebo_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='humanoid_world.sdf',
        description='World file to load'
    )

    # Path to world file
    world_path = PathJoinSubstitution([
        FindPackageShare('humanoid_gazebo'),
        'worlds',
        LaunchConfiguration('world')
    ])

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'),
            '/launch/gz_sim.launch.py'
        ]),
        launch_arguments={'gz_args': ['-r ', world_path]}.items()
    )

    # Spawn humanoid robot
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'humanoid',
            '-topic', 'robot_description',
            '-z', '1.0'
        ],
        output='screen'
    )

    # ROS-Gazebo bridge for communication
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
        ],
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        spawn_robot,
        bridge
    ])
```

### Verifying World Launch

After launching, verify the simulation is running correctly:

```bash
# Check Gazebo topics
gz topic -l

# Echo clock to verify simulation is running
gz topic -e -t /clock

# List spawned models
gz model --list
```

---

## Part 3: Sensor Modeling and Simulation Fidelity

### Sensor Simulation in Gazebo

Accurate sensor modeling is critical for developing perception systems that transfer from simulation to real hardware. Gazebo provides plugins for simulating common robotics sensors with configurable noise models.

### LiDAR Simulation

**LiDAR (Light Detection and Ranging)** sensors measure distances by emitting laser pulses and measuring their return time. In Gazebo, LiDAR is modeled using ray-casting:

```xml
<!-- SDF LiDAR sensor configuration -->
<sensor name="lidar" type="gpu_lidar">
  <pose>0 0 0.5 0 0 0</pose>
  <topic>/humanoid/lidar</topic>
  <update_rate>10</update_rate>
  <lidar>
    <scan>
      <horizontal>
        <samples>640</samples>
        <resolution>1</resolution>
        <min_angle>-1.5708</min_angle>
        <max_angle>1.5708</max_angle>
      </horizontal>
      <vertical>
        <samples>16</samples>
        <resolution>1</resolution>
        <min_angle>-0.26</min_angle>
        <max_angle>0.26</max_angle>
      </vertical>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.02</stddev>
    </noise>
  </lidar>
  <visualize>true</visualize>
</sensor>
```

**LiDAR Parameters for Humanoid Robots:**

| Parameter | Typical Value | Purpose |
|-----------|---------------|---------|
| **Horizontal FOV** | 180° - 360° | Coverage area |
| **Vertical FOV** | 30° - 40° | Multi-layer detection |
| **Range** | 0.1m - 30m | Detection distance |
| **Update Rate** | 10-20 Hz | Scan frequency |
| **Angular Resolution** | 0.1° - 0.5° | Point density |

---

### Depth Camera Simulation

Depth cameras provide per-pixel distance measurements, essential for 3D perception and obstacle avoidance. Gazebo supports both stereo and structured-light depth camera models:

```xml
<!-- SDF Depth Camera configuration -->
<sensor name="depth_camera" type="depth_camera">
  <pose>0.1 0 0.4 0 0 0</pose>
  <topic>/humanoid/depth</topic>
  <update_rate>30</update_rate>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R_FLOAT32</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.005</stddev>
    </noise>
  </camera>
</sensor>
```

**Depth Camera Characteristics:**

| Camera Type | Technology | Strengths | Limitations |
|-------------|------------|-----------|-------------|
| **Stereo** | Dual cameras + matching | Works outdoors | Texture-dependent |
| **Structured Light** | IR pattern projection | High accuracy indoors | Sunlight interference |
| **ToF (Time-of-Flight)** | Phase-shift measurement | Fast, compact | Limited range |

---

### IMU (Inertial Measurement Unit) Simulation

IMUs provide crucial data for humanoid balance and state estimation, measuring linear acceleration and angular velocity:

```xml
<!-- SDF IMU sensor configuration -->
<sensor name="imu" type="imu">
  <pose>0 0 0.3 0 0 0</pose>
  <topic>/humanoid/imu</topic>
  <update_rate>200</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0</mean>
          <stddev>0.0002</stddev>
          <bias_mean>0.00005</bias_mean>
          <bias_stddev>0.00001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0</mean>
          <stddev>0.0002</stddev>
          <bias_mean>0.00005</bias_mean>
          <bias_stddev>0.00001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0</mean>
          <stddev>0.0002</stddev>
          <bias_mean>0.00005</bias_mean>
          <bias_stddev>0.00001</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0</mean>
          <stddev>0.017</stddev>
          <bias_mean>0.001</bias_mean>
          <bias_stddev>0.0001</bias_stddev>
        </noise>
      </x>
      <!-- Similar for y and z axes -->
    </linear_acceleration>
  </imu>
</sensor>
```

**IMU Measurements for Humanoid Control:**

| Measurement | Use in Humanoid | Typical Update Rate |
|-------------|-----------------|-------------------|
| **Angular Velocity** | Balance control, orientation | 200-1000 Hz |
| **Linear Acceleration** | Fall detection, step impact | 200-1000 Hz |
| **Orientation (derived)** | Posture estimation | 100-400 Hz |

---

### The Importance of Sensor Noise Modeling

#### Why Noise Matters

Real sensors produce noisy, imperfect data. Training AI systems on clean, idealized simulation data leads to **overfitting to simulation**—models that fail when deployed on real hardware.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Clean Simulation vs Noisy Reality                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Idealized Simulation                    Real Sensor Data              │
│   ────────────────────                    ────────────────              │
│                                                                          │
│   • Perfect measurements                  • Gaussian noise              │
│   • No sensor drift                       • Bias drift over time        │
│   • Instant response                      • Measurement latency         │
│   • Uniform coverage                      • Dead zones, occlusions      │
│   • Consistent performance                • Temperature-dependent       │
│                                                                          │
│   Result: AI overfits to                  Result: Robust AI that        │
│   perfect data, fails in                  generalizes to real           │
│   real deployment                         hardware conditions           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Noise Types to Model

| Noise Type | Description | Affected Sensors |
|------------|-------------|------------------|
| **Gaussian** | Random variations around true value | All sensors |
| **Bias** | Constant offset from true value | IMU, force sensors |
| **Drift** | Slowly changing bias over time | IMU, encoders |
| **Quantization** | Discrete measurement steps | Encoders, ADCs |
| **Dropout** | Missing measurements | Cameras, LiDAR |
| **Outliers** | Spurious extreme values | LiDAR, depth cameras |

#### Configuring Realistic Noise Models

```python
# Python example: Adding sensor noise for realistic training
import numpy as np
from dataclasses import dataclass

@dataclass
class IMUNoiseModel:
    """Realistic IMU noise parameters based on sensor datasheets."""

    # Gyroscope noise (rad/s)
    gyro_noise_density: float = 0.00017  # rad/s/√Hz
    gyro_random_walk: float = 0.00003    # rad/s²/√Hz
    gyro_bias_stability: float = 0.00005 # rad/s

    # Accelerometer noise (m/s²)
    accel_noise_density: float = 0.002   # m/s²/√Hz
    accel_random_walk: float = 0.0003    # m/s³/√Hz
    accel_bias_stability: float = 0.001  # m/s²

    def apply_noise(self, true_value: np.ndarray, dt: float) -> np.ndarray:
        """Apply realistic noise to IMU measurement."""
        # White noise component
        noise = np.random.normal(0, self.gyro_noise_density / np.sqrt(dt))

        # Bias random walk
        self.current_bias += np.random.normal(0, self.gyro_random_walk * np.sqrt(dt))

        return true_value + noise + self.current_bias


@dataclass
class LiDARNoiseModel:
    """LiDAR noise model with range-dependent characteristics."""

    base_noise: float = 0.01      # Base noise stddev (m)
    range_noise_scale: float = 0.001  # Noise increases with range
    dropout_probability: float = 0.001  # Random point dropout

    def apply_noise(self, ranges: np.ndarray) -> np.ndarray:
        """Apply range-dependent noise and random dropouts."""
        # Range-dependent Gaussian noise
        noise_stddev = self.base_noise + self.range_noise_scale * ranges
        noisy_ranges = ranges + np.random.normal(0, noise_stddev)

        # Random dropouts (set to max range or 0)
        dropout_mask = np.random.random(ranges.shape) < self.dropout_probability
        noisy_ranges[dropout_mask] = 0.0

        return noisy_ranges
```

---

### Unity for High-Fidelity Rendering

While Gazebo excels at physics simulation, **Unity** offers superior visual rendering capabilities that complement physics-focused simulators.

#### Unity's Role in Robotics Simulation

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Gazebo vs Unity: Complementary Strengths                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Gazebo                                  Unity                          │
│   ──────                                  ─────                          │
│                                                                          │
│   ✓ Accurate physics (DART, Bullet)      ✓ Photorealistic rendering     │
│   ✓ Native ROS 2 integration             ✓ Advanced visual effects      │
│   ✓ Robotics-focused sensors             ✓ Large asset ecosystem        │
│   ✓ Open-source                          ✓ Cross-platform deployment    │
│   ✓ Lightweight                          ✓ VR/AR support                │
│                                                                          │
│   Best for: Physics validation,          Best for: Visual perception    │
│   control testing, sensor fusion         training, synthetic data,      │
│                                          human-robot interaction        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Unity Robotics Hub

Unity provides the **Unity Robotics Hub** package for ROS integration:

- **ROS-TCP-Connector**: Bridge between Unity and ROS/ROS 2
- **URDF Importer**: Import robot models from URDF files
- **Sensor Plugins**: Camera, LiDAR, IMU simulation
- **Articulation Bodies**: Physics-based robot joint simulation

#### When to Use Unity

| Use Case | Recommended Platform |
|----------|---------------------|
| Control algorithm development | Gazebo |
| Visual perception training | Unity or Isaac Sim |
| Physics-heavy manipulation | Gazebo |
| Photorealistic synthetic data | Unity or Isaac Sim |
| Human-robot interaction | Unity |
| Real-time visualization | Unity |
| ROS 2 native workflows | Gazebo |

---

## Chapter Summary

This chapter covered the essential concepts of physics simulation for humanoid robotics development:

- **Digital Twins** provide virtual replicas of physical robots, enabling risk-free testing, rapid iteration, and parallel development. Understanding the simulation-reality gap is crucial for successful sim-to-real transfer.

- **Gazebo** serves as the primary physics simulation platform for ROS 2 robotics, offering accurate physics engines (DART, Bullet), comprehensive sensor modeling, and seamless ROS 2 integration.

- **Physics Fundamentals** including gravity, collisions, and friction must be accurately modeled for humanoid robots where balance and ground contact are critical for locomotion.

- **Joint Dynamics** encompassing torque limits, damping, and friction determine how humanoid robots move and respond to control commands.

- **SDF (Simulation Description Format)** extends beyond URDF to define complete simulation worlds including terrain, lighting, and physics properties.

- **Sensor Modeling** with realistic noise characteristics is essential for training AI systems that generalize from simulation to real hardware deployment.

- **Unity** complements Gazebo by providing high-fidelity visual rendering for perception training and synthetic data generation.

Together, these simulation foundations enable the development, testing, and validation of humanoid robot systems before expensive and time-consuming real-world deployment.

---

## Review Questions

Test your understanding of physics simulation and digital twin concepts:

### 1. Simulation-Reality Gap

What is the simulation-reality gap, and why is it particularly challenging for humanoid robots? Describe three techniques used to bridge this gap.

### 2. Physics Engine Selection

Gazebo supports multiple physics engines (DART, Bullet, ODE). What factors would influence your choice of physics engine for simulating a humanoid robot performing manipulation tasks versus locomotion tasks?

### 3. Sensor Noise Importance

Explain why training perception models on noise-free simulated sensor data can lead to poor real-world performance. What types of noise should be modeled for an IMU sensor?

### 4. Key Definitions

> **Digital Twin**: A virtual replica of a physical system that mirrors its real-world counterpart with geometric fidelity, kinematic accuracy, dynamic properties, and sensor simulation. In robotics, digital twins enable risk-free testing, rapid design iteration, and parallel software development by providing an identical control interface for both virtual and physical robots.

> **SDF (Simulation Description Format)**: An XML-based format for describing complete simulation environments in Gazebo, extending beyond robot description to include world properties such as terrain, lighting, physics engine configuration, and multiple robot models. SDF provides native support for sensors, plugins, and nested models that URDF cannot express.
