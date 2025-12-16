---
title: "Chapter 2: URDF Fundamentals for Humanoid Robots"
sidebar_label: "URDF Fundamentals"
sidebar_position: 2
description: Comprehensive guide to URDF structure, links, joints, and robot modeling for humanoid robotics development.
---

# Chapter 2: URDF Fundamentals for Humanoid Robots

## Learning Objectives

- Understand the purpose and structure of URDF (Unified Robot Description Format) in ROS 2
- Define and configure robot links with visual, collision, and inertial properties
- Implement various joint types to connect links and enable robot motion
- Apply URDF concepts to model humanoid robot kinematic chains
- Validate and visualize URDF models using ROS 2 tools

---

## Part 1: Introduction to URDF

### What is URDF?

**URDF (Unified Robot Description Format)** is an XML-based file format used in ROS and ROS 2 to describe the physical structure of a robot. URDF serves as the authoritative specification for a robot's:

- **Kinematic structure**: How links are connected through joints
- **Dynamic properties**: Mass, inertia, and friction characteristics
- **Visual representation**: 3D geometry for rendering and visualization
- **Collision geometry**: Simplified shapes for physics simulation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         URDF in the ROS 2 Ecosystem                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ┌─────────────┐                                │
│                              │  URDF File  │                                │
│                              │  (.urdf)    │                                │
│                              └──────┬──────┘                                │
│                                     │                                        │
│           ┌─────────────────────────┼─────────────────────────┐             │
│           │                         │                         │             │
│           ▼                         ▼                         ▼             │
│   ┌───────────────┐        ┌───────────────┐        ┌───────────────┐      │
│   │    RViz 2     │        │    Gazebo     │        │   MoveIt 2    │      │
│   │               │        │               │        │               │      │
│   │ Visualization │        │   Physics     │        │    Motion     │      │
│   │ & Debugging   │        │  Simulation   │        │   Planning    │      │
│   └───────────────┘        └───────────────┘        └───────────────┘      │
│           │                         │                         │             │
│           └─────────────────────────┼─────────────────────────┘             │
│                                     ▼                                        │
│                        ┌────────────────────────┐                           │
│                        │  robot_state_publisher │                           │
│                        │                        │                           │
│                        │   TF2 Transforms       │                           │
│                        └────────────────────────┘                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why URDF Matters for Humanoid Robots

Humanoid robots are among the most complex robotic systems, featuring:

| Characteristic | Typical Humanoid | URDF Requirement |
|----------------|------------------|------------------|
| **Degrees of Freedom** | 20-50+ joints | Multiple joint definitions |
| **Kinematic Chains** | Arms, legs, torso, head | Hierarchical link structure |
| **Mass Distribution** | Critical for balance | Accurate inertial properties |
| **Collision Avoidance** | Self-collision prevention | Detailed collision geometry |
| **Sensor Mounting** | Cameras, IMUs, force sensors | Fixed links for sensors |

### URDF File Structure Overview

A URDF file follows a strict XML schema with the `<robot>` element as the root:

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">

  <!-- Material definitions (colors) -->
  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>

  <!-- Link definitions (rigid bodies) -->
  <link name="base_link">
    <!-- Visual, collision, inertial properties -->
  </link>

  <link name="torso">
    <!-- ... -->
  </link>

  <!-- Joint definitions (connections) -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
  </joint>

  <!-- Additional links and joints... -->

</robot>
```

### URDF Hierarchy: The Kinematic Tree

URDF models robots as a **kinematic tree**—a hierarchical structure where:

- Each link has exactly **one parent** (except the root)
- Each link can have **multiple children**
- Joints connect parent links to child links
- The tree starts from a **base_link** (root)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Humanoid Robot Kinematic Tree                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              base_link                                       │
│                                  │                                           │
│                            [fixed joint]                                     │
│                                  │                                           │
│                               torso                                          │
│                    ┌─────────────┼─────────────┐                            │
│              [revolute]     [revolute]    [revolute]                        │
│                    │             │             │                             │
│                  head      left_shoulder  right_shoulder                    │
│                    │             │             │                             │
│              [revolute]    [revolute]    [revolute]                         │
│                    │             │             │                             │
│                head_pan    left_upper_arm  right_upper_arm                  │
│                                  │             │                             │
│                            [revolute]    [revolute]                         │
│                                  │             │                             │
│                            left_forearm   right_forearm                     │
│                                  │             │                             │
│                            [revolute]    [revolute]                         │
│                                  │             │                             │
│                             left_hand     right_hand                        │
│                                                                              │
│                    (Legs follow similar pattern)                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Understanding Links

### What is a Link?

A **link** represents a rigid body in the robot model. In physical terms, a link is a solid component that does not deform during operation. For a humanoid robot, links include:

- Torso/chest
- Head
- Upper arms, forearms, hands
- Thighs, shins, feet
- Pelvis/hip structure

### Link Components

Each link in URDF can contain three optional elements:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Link Components                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   <link name="left_thigh">                                                  │
│   │                                                                          │
│   ├── <visual>          ─────▶  What you SEE in RViz/Gazebo                │
│   │   └── geometry, material, origin                                        │
│   │                                                                          │
│   ├── <collision>       ─────▶  What PHYSICS engines use                   │
│   │   └── geometry, origin                                                  │
│   │                                                                          │
│   └── <inertial>        ─────▶  Mass properties for DYNAMICS               │
│       └── mass, inertia, origin                                             │
│                                                                              │
│   </link>                                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Visual Properties

The `<visual>` element defines how the link appears in visualization tools:

```xml
<link name="torso">
  <visual>
    <!-- Position/orientation relative to link origin -->
    <origin xyz="0 0 0.2" rpy="0 0 0"/>

    <!-- 3D shape -->
    <geometry>
      <box size="0.3 0.2 0.4"/>
    </geometry>

    <!-- Appearance -->
    <material name="gray">
      <color rgba="0.5 0.5 0.5 1.0"/>
    </material>
  </visual>
</link>
```

**Available Geometry Types:**

| Geometry | Parameters | Best For |
|----------|------------|----------|
| `<box>` | `size="x y z"` | Torso, rectangular parts |
| `<cylinder>` | `radius="r" length="l"` | Limb segments |
| `<sphere>` | `radius="r"` | Joints, rounded parts |
| `<mesh>` | `filename="path"` | Detailed custom shapes |

### Collision Properties

The `<collision>` element defines the geometry used for physics simulation and collision detection:

```xml
<link name="torso">
  <!-- Visual: detailed mesh for appearance -->
  <visual>
    <geometry>
      <mesh filename="package://humanoid/meshes/torso_detailed.dae"/>
    </geometry>
  </visual>

  <!-- Collision: simplified geometry for performance -->
  <collision>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <geometry>
      <box size="0.3 0.2 0.4"/>
    </geometry>
  </collision>
</link>
```

**Best Practices for Collision Geometry:**

| Guideline | Reason |
|-----------|--------|
| Use simpler shapes than visual | Faster collision computation |
| Slightly larger than visual | Safety margin for contact |
| Convex shapes preferred | More stable physics simulation |
| Multiple collision elements allowed | Complex shapes from primitives |

### Inertial Properties

The `<inertial>` element defines mass properties critical for dynamics simulation:

```xml
<link name="left_thigh">
  <inertial>
    <!-- Center of mass position -->
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>

    <!-- Total mass in kg -->
    <mass value="3.5"/>

    <!-- Inertia tensor (kg·m²) -->
    <inertia
      ixx="0.05" ixy="0.0"  ixz="0.0"
                 iyy="0.05" iyz="0.0"
                            izz="0.01"/>
  </inertial>
</link>
```

**Understanding the Inertia Tensor:**

The inertia tensor is a 3×3 symmetric matrix describing how mass is distributed:

$$
I = \begin{bmatrix}
I_{xx} & I_{xy} & I_{xz} \\
I_{xy} & I_{yy} & I_{yz} \\
I_{xz} & I_{yz} & I_{zz}
\end{bmatrix}
$$

| Component | Physical Meaning |
|-----------|------------------|
| $I_{xx}$ | Resistance to rotation about X-axis |
| $I_{yy}$ | Resistance to rotation about Y-axis |
| $I_{zz}$ | Resistance to rotation about Z-axis |
| $I_{xy}, I_{xz}, I_{yz}$ | Coupling between rotation axes |

### Complete Link Example

```xml
<!-- Humanoid upper arm link with all properties -->
<link name="left_upper_arm">

  <!-- Visual representation -->
  <visual>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.04" length="0.30"/>
    </geometry>
    <material name="skin_tone">
      <color rgba="0.9 0.75 0.65 1.0"/>
    </material>
  </visual>

  <!-- Collision geometry -->
  <collision>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.045" length="0.30"/>
    </geometry>
  </collision>

  <!-- Dynamic properties -->
  <inertial>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <mass value="2.0"/>
    <inertia
      ixx="0.015" ixy="0.0"   ixz="0.0"
                  iyy="0.015" iyz="0.0"
                              izz="0.002"/>
  </inertial>

</link>
```

---

## Understanding Joints

### What is a Joint?

A **joint** defines the kinematic relationship between two links, specifying:

- **Parent link**: The reference frame
- **Child link**: The moving frame
- **Joint type**: The allowed motion
- **Axis**: Direction of motion (for moving joints)
- **Limits**: Constraints on motion range

### Joint Types in URDF

URDF supports six joint types to model different mechanical connections:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         URDF Joint Types                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Type         │ DOF │ Motion              │ Humanoid Application            │
│  ─────────────┼─────┼─────────────────────┼────────────────────────────────│
│  fixed        │  0  │ No relative motion  │ Sensor mounts, rigid parts     │
│  revolute     │  1  │ Rotation with limits│ Elbow, knee, ankle             │
│  continuous   │  1  │ Unlimited rotation  │ Wheel joints (rare in humanoid)│
│  prismatic    │  1  │ Linear translation  │ Telescoping mechanisms         │
│  floating     │  6  │ Free 6-DOF motion   │ Base link (implicit)           │
│  planar       │  3  │ Motion in XY plane  │ Specialized applications       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fixed Joints

Fixed joints rigidly connect two links with no relative motion:

```xml
<joint name="head_camera_mount" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.1" rpy="0 0 0"/>
</joint>
```

**Use Cases:**
- Sensor attachments
- Rigid structural connections
- Combining URDF modules

### Revolute Joints

Revolute joints allow rotation about a single axis within defined limits—the most common joint type for humanoids:

```xml
<joint name="left_elbow" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_forearm"/>

  <!-- Joint position relative to parent -->
  <origin xyz="0 0 -0.30" rpy="0 0 0"/>

  <!-- Rotation axis (in joint frame) -->
  <axis xyz="0 1 0"/>

  <!-- Motion constraints -->
  <limit
    lower="0.0"        <!-- Minimum angle (rad) -->
    upper="2.5"        <!-- Maximum angle (rad) -->
    effort="100.0"     <!-- Maximum torque (Nm) -->
    velocity="3.0"     <!-- Maximum velocity (rad/s) -->
  />

  <!-- Dynamic properties -->
  <dynamics
    damping="0.5"      <!-- Viscous damping -->
    friction="0.1"     <!-- Coulomb friction -->
  />
</joint>
```

### Continuous Joints

Continuous joints allow unlimited rotation (no position limits):

```xml
<joint name="wheel_joint" type="continuous">
  <parent link="leg"/>
  <child link="wheel"/>
  <origin xyz="0 0 -0.5" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit effort="50.0" velocity="10.0"/>
</joint>
```

### Prismatic Joints

Prismatic joints allow linear translation along an axis:

```xml
<joint name="gripper_slide" type="prismatic">
  <parent link="gripper_base"/>
  <child link="gripper_finger"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit
    lower="0.0"
    upper="0.05"
    effort="20.0"
    velocity="0.1"
  />
</joint>
```

### Joint Axis Definition

The `<axis>` element specifies the direction of motion in the joint frame:

| Axis | Vector | Motion |
|------|--------|--------|
| X-axis | `xyz="1 0 0"` | Roll (for revolute) |
| Y-axis | `xyz="0 1 0"` | Pitch (for revolute) |
| Z-axis | `xyz="0 0 1"` | Yaw (for revolute) |

### Joint Limits

Proper joint limits are essential for humanoid safety and realistic simulation:

```xml
<limit
  lower="-1.57"      <!-- -90 degrees in radians -->
  upper="1.57"       <!-- +90 degrees in radians -->
  effort="150.0"     <!-- Maximum torque in Nm -->
  velocity="5.0"     <!-- Maximum angular velocity in rad/s -->
/>
```

**Humanoid Joint Limit Guidelines:**

| Joint | Typical Range | Effort (Nm) |
|-------|---------------|-------------|
| Hip pitch | -30° to +120° | 100-200 |
| Hip roll | -45° to +45° | 80-150 |
| Hip yaw | -45° to +45° | 60-100 |
| Knee | 0° to +150° | 100-200 |
| Ankle pitch | -45° to +45° | 50-100 |
| Ankle roll | -30° to +30° | 40-80 |
| Shoulder pitch | -180° to +60° | 50-100 |
| Elbow | 0° to +145° | 30-60 |

### Complete Joint Example: Hip Joint Complex

Humanoid hip joints typically require 3 DOF (yaw, roll, pitch), modeled as three sequential revolute joints:

```xml
<!-- Hip Yaw (rotation about vertical axis) -->
<joint name="left_hip_yaw" type="revolute">
  <parent link="pelvis"/>
  <child link="left_hip_yaw_link"/>
  <origin xyz="0.1 0.1 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.78" upper="0.78" effort="100" velocity="4"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>

<link name="left_hip_yaw_link">
  <inertial>
    <mass value="0.5"/>
    <inertia ixx="0.001" iyy="0.001" izz="0.001"
             ixy="0" ixz="0" iyz="0"/>
  </inertial>
</link>

<!-- Hip Roll (rotation about forward axis) -->
<joint name="left_hip_roll" type="revolute">
  <parent link="left_hip_yaw_link"/>
  <child link="left_hip_roll_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.52" upper="0.52" effort="120" velocity="4"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>

<link name="left_hip_roll_link">
  <inertial>
    <mass value="0.5"/>
    <inertia ixx="0.001" iyy="0.001" izz="0.001"
             ixy="0" ixz="0" iyz="0"/>
  </inertial>
</link>

<!-- Hip Pitch (rotation about lateral axis) -->
<joint name="left_hip_pitch" type="revolute">
  <parent link="left_hip_roll_link"/>
  <child link="left_thigh"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.52" upper="2.09" effort="150" velocity="4"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>
```

### Visualizing Links and Joints

Use ROS 2 tools to validate your URDF structure:

```bash
# Check URDF syntax
ros2 run urdf_parser_plugin check_urdf humanoid.urdf

# View robot model in RViz
ros2 launch urdf_tutorial display.launch.py model:=humanoid.urdf

# Print kinematic tree structure
ros2 run urdfdom urdf_to_graphviz humanoid.urdf
```

```python
# Python: Load and inspect URDF
from urdf_parser_py.urdf import URDF

robot = URDF.from_xml_file('humanoid.urdf')

print(f"Robot name: {robot.name}")
print(f"Number of links: {len(robot.links)}")
print(f"Number of joints: {len(robot.joints)}")

# List all joints
for joint in robot.joints:
    print(f"Joint: {joint.name}, Type: {joint.type}, "
          f"Parent: {joint.parent}, Child: {joint.child}")
```

---

## Part 2: Kinematics and XML Configuration

### Understanding Robot Kinematics

Kinematics is the study of motion without considering the forces that cause it. For humanoid robots, kinematics determines the relationship between joint angles and the position/orientation of body parts like hands and feet.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Kinematics in Humanoid Robotics                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Joint Space (θ)                              Task Space (x, y, z)         │
│   ───────────────                              ─────────────────────         │
│                                                                              │
│   θ₁ = shoulder_pitch                          Hand Position:               │
│   θ₂ = shoulder_roll          ═══════════▶     x = 0.45 m                   │
│   θ₃ = shoulder_yaw           Forward          y = 0.30 m                   │
│   θ₄ = elbow                  Kinematics       z = 1.10 m                   │
│   θ₅ = wrist_pitch                                                          │
│   θ₆ = wrist_roll                              Orientation:                 │
│                               ◀═══════════     roll, pitch, yaw             │
│                               Inverse                                        │
│                               Kinematics                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Forward Kinematics

**Forward Kinematics (FK)** computes the position and orientation of the robot's end-effector (e.g., hand, foot) given all joint angles. This is a straightforward mathematical problem with a unique solution.

#### Mathematical Foundation

For a serial kinematic chain, forward kinematics uses transformation matrices:

$$
T_{0}^{n} = T_{0}^{1} \cdot T_{1}^{2} \cdot T_{2}^{3} \cdots T_{n-1}^{n}
$$

Where each $T_{i-1}^{i}$ represents the transformation from link $i-1$ to link $i$, computed from:
- Joint angle $\theta_i$ (for revolute joints)
- Link length and offset parameters

#### Forward Kinematics Process

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Forward Kinematics Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input: Joint Angles                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  θ = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆]  (e.g., arm joint configuration)    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  For each joint i:                                                   │   │
│   │    1. Read URDF: origin, axis, joint type                           │   │
│   │    2. Compute local transform T_{i-1}^{i}(θᵢ)                       │   │
│   │    3. Multiply: T_{0}^{i} = T_{0}^{i-1} · T_{i-1}^{i}               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                               │
│                              ▼                                               │
│   Output: End-Effector Pose                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Position: (x, y, z)     Orientation: (roll, pitch, yaw)            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Forward Kinematics in ROS 2

```python
# Forward kinematics using ROS 2 TF2
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

class ForwardKinematicsNode(Node):
    def __init__(self):
        super().__init__('forward_kinematics')

        # TF2 buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer to compute FK periodically
        self.timer = self.create_timer(0.1, self.compute_fk)

    def compute_fk(self):
        """Compute forward kinematics using TF2 transforms."""
        try:
            # Get transform from base to left hand
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'base_link',      # Target frame
                'left_hand',      # Source frame
                rclpy.time.Time() # Latest available
            )

            # Extract position
            pos = transform.transform.translation
            self.get_logger().info(
                f'Left hand position: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})'
            )

        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}')
```

---

### Inverse Kinematics

**Inverse Kinematics (IK)** solves the opposite problem: given a desired end-effector position and orientation, find the joint angles that achieve it. This is significantly more complex than forward kinematics.

#### IK Challenges

| Challenge | Description | Humanoid Impact |
|-----------|-------------|-----------------|
| **Multiple Solutions** | Many joint configurations reach same pose | Must choose appropriate solution |
| **No Solution** | Target outside reachable workspace | Need workspace analysis |
| **Singularities** | Configurations with reduced mobility | Avoid during motion planning |
| **Redundancy** | More DOF than needed for task | Enables optimization |

#### Inverse Kinematics Methods

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Inverse Kinematics Methods                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Method              │ Approach           │ Pros/Cons                       │
│  ────────────────────┼────────────────────┼────────────────────────────────│
│                      │                    │                                 │
│  Analytical/         │ Closed-form        │ ✓ Fast, exact                  │
│  Geometric           │ equations          │ ✗ Only for specific structures │
│                      │                    │                                 │
│  Jacobian            │ Iterative          │ ✓ General purpose              │
│  (Pseudo-inverse)    │ linearization      │ ✗ May not converge             │
│                      │                    │                                 │
│  Numerical           │ Optimization       │ ✓ Handles constraints          │
│  Optimization        │ (gradient descent) │ ✗ Computationally expensive    │
│                      │                    │                                 │
│  Machine Learning    │ Neural network     │ ✓ Very fast inference          │
│                      │ approximation      │ ✗ Requires training data       │
│                      │                    │                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Jacobian-Based IK

The Jacobian matrix relates joint velocities to end-effector velocities:

$$
\dot{x} = J(\theta) \cdot \dot{\theta}
$$

For IK, we invert this relationship:

$$
\dot{\theta} = J^{\dagger}(\theta) \cdot \dot{x}
$$

Where $J^{\dagger}$ is the pseudo-inverse of the Jacobian.

```python
# Conceptual Jacobian IK implementation
import numpy as np

class JacobianIK:
    """Jacobian-based inverse kinematics solver."""

    def __init__(self, robot_model):
        self.model = robot_model
        self.max_iterations = 100
        self.tolerance = 1e-4

    def solve(self, target_pose, initial_joints):
        """
        Solve IK using damped least squares (Levenberg-Marquardt).

        Args:
            target_pose: Desired end-effector pose (position + orientation)
            initial_joints: Starting joint configuration

        Returns:
            Joint angles achieving target pose, or None if failed
        """
        joints = np.array(initial_joints)
        damping = 0.01  # Damping factor for stability

        for iteration in range(self.max_iterations):
            # Compute current end-effector pose via FK
            current_pose = self.model.forward_kinematics(joints)

            # Compute pose error
            error = self.compute_pose_error(target_pose, current_pose)

            # Check convergence
            if np.linalg.norm(error) < self.tolerance:
                return joints

            # Compute Jacobian at current configuration
            J = self.model.compute_jacobian(joints)

            # Damped least squares solution
            # Δθ = Jᵀ(JJᵀ + λ²I)⁻¹ · error
            JJT = J @ J.T
            delta_joints = J.T @ np.linalg.solve(
                JJT + damping**2 * np.eye(JJT.shape[0]),
                error
            )

            # Update joints
            joints = joints + delta_joints

            # Apply joint limits
            joints = self.model.clamp_to_limits(joints)

        return None  # Failed to converge

    def compute_pose_error(self, target, current):
        """Compute 6D pose error (position + orientation)."""
        position_error = target[:3] - current[:3]
        orientation_error = self.orientation_error(target[3:], current[3:])
        return np.concatenate([position_error, orientation_error])
```

---

### Joint Types for Humanoid Robots

Different joints serve different purposes in humanoid robot design. Understanding when to use each type is crucial for accurate URDF modeling.

#### Detailed Joint Type Reference

| Joint Type | Symbol | DOF | Motion Description | URDF Element |
|------------|--------|-----|-------------------|--------------|
| **Fixed** | ─ | 0 | No motion; rigid connection | `type="fixed"` |
| **Revolute** | ⟳ | 1 | Bounded rotation about axis | `type="revolute"` |
| **Continuous** | ⟳∞ | 1 | Unbounded rotation | `type="continuous"` |
| **Prismatic** | ↔ | 1 | Linear sliding motion | `type="prismatic"` |
| **Floating** | ✦ | 6 | Free motion in space | `type="floating"` |
| **Planar** | ⊞ | 3 | Motion in a plane (x, y, θ) | `type="planar"` |

#### Humanoid Joint Configuration

A typical humanoid robot uses joints in the following pattern:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Humanoid Robot Joint Configuration                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                            HEAD (2 DOF)                                      │
│                         ┌─────────────┐                                     │
│                         │  neck_pitch │ revolute                            │
│                         │  neck_yaw   │ revolute                            │
│                         └─────────────┘                                     │
│                                │                                             │
│         LEFT ARM (7 DOF)       │        RIGHT ARM (7 DOF)                   │
│        ┌──────────────┐        │       ┌──────────────┐                     │
│        │shoulder_pitch│        │       │shoulder_pitch│                     │
│        │shoulder_roll │  ┌─────┴─────┐ │shoulder_roll │                     │
│        │shoulder_yaw  │  │   TORSO   │ │shoulder_yaw  │                     │
│        │elbow_pitch   │  │  (fixed)  │ │elbow_pitch   │                     │
│        │elbow_yaw     │  └─────┬─────┘ │elbow_yaw     │                     │
│        │wrist_pitch   │        │       │wrist_pitch   │                     │
│        │wrist_roll    │        │       │wrist_roll    │                     │
│        └──────────────┘        │       └──────────────┘                     │
│                                │                                             │
│                         ┌─────┴─────┐                                       │
│                         │  PELVIS   │                                       │
│                         └─────┬─────┘                                       │
│                    ┌──────────┴──────────┐                                  │
│         LEFT LEG (6 DOF)        RIGHT LEG (6 DOF)                           │
│        ┌──────────────┐        ┌──────────────┐                             │
│        │hip_yaw       │        │hip_yaw       │                             │
│        │hip_roll      │        │hip_roll      │                             │
│        │hip_pitch     │        │hip_pitch     │                             │
│        │knee_pitch    │        │knee_pitch    │                             │
│        │ankle_pitch   │        │ankle_pitch   │                             │
│        │ankle_roll    │        │ankle_roll    │                             │
│        └──────────────┘        └──────────────┘                             │
│                                                                              │
│        Total: 2 + 7 + 7 + 6 + 6 = 28 DOF (typical humanoid)                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### URDF XML Configuration Example

The following complete example demonstrates two links connected by a revolute joint, suitable for modeling a humanoid arm segment:

```xml
<?xml version="1.0"?>
<robot name="arm_segment" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Material Definitions -->
  <material name="aluminum">
    <color rgba="0.8 0.8 0.85 1.0"/>
  </material>

  <material name="dark_gray">
    <color rgba="0.3 0.3 0.3 1.0"/>
  </material>

  <!-- ============================================ -->
  <!-- LINK 1: Upper Arm                            -->
  <!-- ============================================ -->
  <link name="upper_arm">

    <!-- Visual: What you see in RViz/Gazebo -->
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.30"/>
      </geometry>
      <material name="aluminum"/>
    </visual>

    <!-- Collision: Used by physics engine -->
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.042" length="0.30"/>
      </geometry>
    </collision>

    <!-- Inertial: Mass properties for dynamics -->
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia
        ixx="0.015" ixy="0.0"   ixz="0.0"
                    iyy="0.015" iyz="0.0"
                                izz="0.002"/>
    </inertial>

  </link>

  <!-- ============================================ -->
  <!-- LINK 2: Forearm                              -->
  <!-- ============================================ -->
  <link name="forearm">

    <visual>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.035" length="0.25"/>
      </geometry>
      <material name="aluminum"/>
    </visual>

    <collision>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.037" length="0.25"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia
        ixx="0.008" ixy="0.0"   ixz="0.0"
                    iyy="0.008" iyz="0.0"
                                izz="0.001"/>
    </inertial>

  </link>

  <!-- ============================================ -->
  <!-- JOINT: Elbow (Revolute)                      -->
  <!-- Connects upper_arm to forearm                -->
  <!-- ============================================ -->
  <joint name="elbow_joint" type="revolute">

    <!-- Parent and child links -->
    <parent link="upper_arm"/>
    <child link="forearm"/>

    <!-- Joint origin: position relative to parent link -->
    <!-- Located at the bottom of the upper arm -->
    <origin xyz="0 0 -0.30" rpy="0 0 0"/>

    <!-- Rotation axis: Y-axis for pitch motion -->
    <axis xyz="0 1 0"/>

    <!-- Joint limits -->
    <limit
      lower="0.0"           <!-- Fully extended (0 rad) -->
      upper="2.53"          <!-- Fully flexed (~145 degrees) -->
      effort="60.0"         <!-- Max torque: 60 Nm -->
      velocity="3.14"       <!-- Max velocity: ~180 deg/s -->
    />

    <!-- Dynamic properties -->
    <dynamics
      damping="0.5"         <!-- Viscous friction coefficient -->
      friction="0.1"        <!-- Coulomb friction -->
    />

    <!-- Safety controller (optional, for ros2_control) -->
    <safety_controller
      soft_lower_limit="0.05"
      soft_upper_limit="2.48"
      k_position="100"
      k_velocity="10"
    />

  </joint>

</robot>
```

### Key Elements Explained

| Element | Purpose | Example Value |
|---------|---------|---------------|
| `<origin xyz="..." rpy="..."/>` | Position/rotation relative to parent | `xyz="0 0 -0.30"` = 30cm below |
| `<axis xyz="..."/>` | Rotation/translation direction | `xyz="0 1 0"` = Y-axis (pitch) |
| `<limit lower="..." upper="..."/>` | Motion range in radians | `0.0` to `2.53` rad |
| `<limit effort="..."/>` | Maximum force/torque | `60.0` Nm |
| `<limit velocity="..."/>` | Maximum joint speed | `3.14` rad/s |
| `<dynamics damping="..."/>` | Viscous friction | `0.5` Nm·s/rad |
| `<dynamics friction="..."/>` | Coulomb friction | `0.1` Nm |

### Validating the URDF

After creating your URDF, validate it using ROS 2 tools:

```bash
# Check URDF syntax and structure
ros2 run urdf_parser_plugin check_urdf arm_segment.urdf

# Expected output:
# robot name is: arm_segment
# ---------- Successfully Parsed XML ---------------
# root Link: upper_arm has 1 child(ren)
#     child(1):  forearm

# Visualize in RViz2
ros2 launch urdf_tutorial display.launch.py model:=arm_segment.urdf

# Generate PDF visualization of kinematic tree
ros2 run urdfdom urdf_to_graphviz arm_segment.urdf
evince arm_segment.pdf
```

---

## Part 3: Transmissions and ros2_control Integration

### Why Transmissions Are Necessary

While URDF defines the physical structure of a robot (links, joints, dynamics), it doesn't specify how joints connect to actual motor controllers. The **`<transmission>`** element bridges this gap, enabling ros2_control to interface with simulated or real actuators.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    URDF + Transmissions + ros2_control                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   URDF Definition              Transmission              ros2_control        │
│   ───────────────              ────────────              ────────────        │
│                                                                              │
│   ┌─────────────┐             ┌─────────────┐          ┌─────────────┐      │
│   │   <joint>   │────────────▶│<transmission>│─────────▶│  Hardware   │      │
│   │             │             │             │          │  Interface  │      │
│   │ - name      │             │ - type      │          │             │      │
│   │ - type      │             │ - joint     │          │ - command   │      │
│   │ - limits    │             │ - actuator  │          │ - state     │      │
│   └─────────────┘             └─────────────┘          └──────┬──────┘      │
│                                                               │             │
│                                                               ▼             │
│                                                        ┌─────────────┐      │
│                                                        │ Controllers │      │
│                                                        │             │      │
│                                                        │ - position  │      │
│                                                        │ - velocity  │      │
│                                                        │ - effort    │      │
│                                                        └─────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Role of Transmissions

Transmissions define:

| Aspect | Description | Example |
|--------|-------------|---------|
| **Joint-Actuator Mapping** | Which actuator drives which joint | Motor 1 → left_knee |
| **Mechanical Reduction** | Gear ratios between motor and joint | 100:1 reduction |
| **Interface Type** | How the controller communicates | Position, velocity, or effort |
| **Multiple Actuators** | Differential drives, coupled joints | Two motors driving one joint |

### Transmission Syntax

```xml
<!-- Basic transmission for a single joint -->
<transmission name="left_knee_transmission">
  <type>transmission_interface/SimpleTransmission</type>

  <joint name="left_knee">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>

  <actuator name="left_knee_motor">
    <mechanicalReduction>100</mechanicalReduction>
  </actuator>
</transmission>
```

### Transmission Types

| Transmission Type | Use Case | Description |
|-------------------|----------|-------------|
| **SimpleTransmission** | Most joints | 1:1 joint-to-actuator mapping with optional gear ratio |
| **DifferentialTransmission** | Wrist, ankle | Two actuators drive two joints in differential configuration |
| **FourBarLinkageTransmission** | Parallel mechanisms | Complex mechanical linkages |

### ros2_control Integration

In ROS 2, transmissions work with the ros2_control framework. The modern approach uses a `<ros2_control>` tag:

```xml
<!-- ros2_control configuration for humanoid leg -->
<ros2_control name="HumanoidLegSystem" type="system">

  <hardware>
    <plugin>gazebo_ros2_control/GazeboSystem</plugin>
  </hardware>

  <!-- Hip Pitch Joint -->
  <joint name="left_hip_pitch">
    <command_interface name="position">
      <param name="min">-0.52</param>
      <param name="max">2.09</param>
    </command_interface>
    <command_interface name="velocity">
      <param name="min">-4.0</param>
      <param name="max">4.0</param>
    </command_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
    <state_interface name="effort"/>
  </joint>

  <!-- Knee Joint -->
  <joint name="left_knee">
    <command_interface name="position">
      <param name="min">0.0</param>
      <param name="max">2.5</param>
    </command_interface>
    <command_interface name="velocity">
      <param name="min">-5.0</param>
      <param name="max">5.0</param>
    </command_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
    <state_interface name="effort"/>
  </joint>

  <!-- Ankle Pitch Joint -->
  <joint name="left_ankle_pitch">
    <command_interface name="position">
      <param name="min">-0.78</param>
      <param name="max">0.78</param>
    </command_interface>
    <state_interface name="position"/>
    <state_interface name="velocity"/>
  </joint>

</ros2_control>
```

### Hardware Interfaces

| Interface Type | Description | Humanoid Use Case |
|----------------|-------------|-------------------|
| **PositionJointInterface** | Commands target position | Precise pose control |
| **VelocityJointInterface** | Commands target velocity | Smooth motion |
| **EffortJointInterface** | Commands torque/force | Compliant control, force feedback |

### Complete Transmission Example for Humanoid Arm

```xml
<?xml version="1.0"?>
<robot name="humanoid_arm" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- ... links and joints defined above ... -->

  <!-- ============================================ -->
  <!-- TRANSMISSIONS                                -->
  <!-- ============================================ -->

  <!-- Shoulder Pitch Transmission -->
  <transmission name="shoulder_pitch_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder_pitch">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="shoulder_pitch_motor">
      <mechanicalReduction>100</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Shoulder Roll Transmission -->
  <transmission name="shoulder_roll_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_shoulder_roll">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="shoulder_roll_motor">
      <mechanicalReduction>100</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Elbow Transmission -->
  <transmission name="elbow_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_elbow">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="elbow_motor">
      <mechanicalReduction>50</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- ============================================ -->
  <!-- ROS2_CONTROL CONFIGURATION                   -->
  <!-- ============================================ -->

  <ros2_control name="HumanoidArmControl" type="system">
    <hardware>
      <plugin>gazebo_ros2_control/GazeboSystem</plugin>
    </hardware>

    <joint name="left_shoulder_pitch">
      <command_interface name="effort"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>

    <joint name="left_shoulder_roll">
      <command_interface name="effort"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
      <state_interface name="effort"/>
    </joint>

    <joint name="left_elbow">
      <command_interface name="position"/>
      <state_interface name="position"/>
      <state_interface name="velocity"/>
    </joint>
  </ros2_control>

</robot>
```

### Controller Configuration (YAML)

The controller manager uses YAML configuration to load and configure controllers:

```yaml
# humanoid_controllers.yaml
controller_manager:
  ros__parameters:
    update_rate: 500  # Hz

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    arm_controller:
      type: joint_trajectory_controller/JointTrajectoryController

arm_controller:
  ros__parameters:
    joints:
      - left_shoulder_pitch
      - left_shoulder_roll
      - left_elbow

    command_interfaces:
      - position

    state_interfaces:
      - position
      - velocity

    state_publish_rate: 100.0
    action_monitor_rate: 20.0

    constraints:
      stopped_velocity_tolerance: 0.01
      goal_time: 0.0
```

### Launching Controllers

```bash
# Load and start controllers
ros2 control load_controller --set-state active joint_state_broadcaster
ros2 control load_controller --set-state active arm_controller

# List active controllers
ros2 control list_controllers

# Check hardware interfaces
ros2 control list_hardware_interfaces
```

---

## Chapter Summary

This chapter provided a comprehensive foundation in URDF for humanoid robot modeling:

- **URDF Structure**: We explored how URDF uses XML to define robots as kinematic trees, with the `<robot>` element containing `<link>` and `<joint>` definitions that specify the complete physical structure.

- **Links**: Each link represents a rigid body with three key components—`<visual>` for appearance, `<collision>` for physics simulation, and `<inertial>` for dynamic properties including mass and inertia tensors.

- **Joints**: Six joint types (fixed, revolute, continuous, prismatic, floating, planar) enable modeling of any mechanical connection, with revolute joints being most common in humanoids for elbow, knee, and ankle articulation.

- **Kinematics**: Forward kinematics computes end-effector pose from joint angles using transformation matrices, while inverse kinematics solves the reverse problem using methods like Jacobian pseudo-inverse iteration.

- **Joint Configuration**: A typical humanoid robot requires 28+ degrees of freedom distributed across head (2), arms (7 each), and legs (6 each), modeled as chains of revolute joints.

- **Transmissions**: The `<transmission>` element and ros2_control `<ros2_control>` tags connect URDF joints to hardware interfaces, enabling position, velocity, or effort control through the ros2_control framework.

Mastering URDF is essential for humanoid robotics development, as it serves as the foundation for simulation in Gazebo, visualization in RViz, motion planning with MoveIt 2, and real robot control through ros2_control.

---

## Review Questions

Test your understanding of URDF elements and structure:

### 1. Link Components

A URDF link contains three optional child elements: `<visual>`, `<collision>`, and `<inertial>`. Explain the purpose of each and why you might use different geometries for visual and collision elements.

### 2. Joint Configuration

A humanoid robot's hip joint requires 3 degrees of freedom (yaw, roll, pitch). Explain how this would be modeled in URDF, and why multiple sequential revolute joints are used instead of a single ball joint.

### 3. Transmission Purpose

Why are `<transmission>` elements necessary in a URDF file? What happens if you define joints without transmissions when using ros2_control?

### 4. Key Definitions

> **Link**: A rigid body element in URDF that represents a physical component of the robot. Each link can contain `<visual>` geometry for rendering, `<collision>` geometry for physics simulation, and `<inertial>` properties (mass, center of mass, inertia tensor) for dynamic simulation. Links form the nodes of the robot's kinematic tree structure.

> **Joint**: A URDF element that defines the kinematic relationship between two links (parent and child), specifying the type of allowed motion (fixed, revolute, continuous, prismatic, floating, or planar), the axis of motion, position/velocity/effort limits, and dynamic properties such as damping and friction. Joints form the edges connecting links in the kinematic tree.
