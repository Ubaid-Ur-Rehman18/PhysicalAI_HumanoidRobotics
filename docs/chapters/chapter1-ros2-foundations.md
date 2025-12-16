---
title: "Chapter 1: ROS 2 Foundations for Humanoid Robotics"
sidebar_position: 1
description: ROS 2 architecture, rclpy basics, and introduction to URDF for humanoid robot development.
---

# Chapter 1: ROS 2 Foundations for Humanoid Robotics

## Learning Objectives

- Understand the ROS 2 communication architecture (nodes, topics, services, actions)
- Learn to create ROS 2 applications using rclpy (Python client library)
- Grasp the fundamentals of URDF for robot description
- Apply these concepts to humanoid robot development

---

## Part 3: Introduction to URDF

### What is URDF?

The **Unified Robot Description Format (URDF)** is an XML-based file format used in ROS and ROS 2 to describe the complete physical structure of a robot. For humanoid robotics, URDF serves as the authoritative definition of the robot's mechanical design, enabling simulation, visualization, and motion planning.

### Why URDF Matters for Humanoid Robots

Humanoid robots are among the most complex robotic systems, featuring:

- Multiple degrees of freedom (typically 20-50+ joints)
- Complex kinematic chains (arms, legs, torso, head)
- Varied joint types and motion constraints
- Sophisticated sensor placements

URDF provides a standardized way to capture all this complexity in a machine-readable format that integrates seamlessly with the ROS 2 ecosystem.

### Core URDF Elements

#### Links

Links represent the rigid bodies of the robot. For a humanoid, typical links include:

```xml
<link name="torso">
  <visual>
    <geometry>
      <box size="0.3 0.2 0.4"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="0.3 0.2 0.4"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="10.0"/>
    <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
  </inertial>
</link>
```

Each link defines:
- **Visual**: The graphical representation for RViz and Gazebo rendering
- **Collision**: Simplified geometry for physics collision detection
- **Inertial**: Mass properties for dynamic simulation

#### Joints

Joints connect links and define their relative motion:

```xml
<joint name="left_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="left_upper_arm"/>
  <origin xyz="0.15 0.1 0.15" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
</joint>
```

Common joint types for humanoids:
- **revolute**: Rotational joint with limits (most arm/leg joints)
- **continuous**: Unlimited rotation (wheel joints)
- **prismatic**: Linear motion (telescoping mechanisms)
- **fixed**: No relative motion (rigid attachments)

#### Complete Humanoid Structure

A typical humanoid URDF defines the following kinematic chains:

| Chain | Links | Joints | Purpose |
|-------|-------|--------|---------|
| Torso | pelvis, waist, chest | 2-3 | Core stability and posture |
| Head | neck, head | 2 | Vision and perception |
| Left Arm | shoulder, upper_arm, forearm, hand | 6-7 | Manipulation |
| Right Arm | shoulder, upper_arm, forearm, hand | 6-7 | Manipulation |
| Left Leg | hip, thigh, shin, foot | 6 | Locomotion |
| Right Leg | hip, thigh, shin, foot | 6 | Locomotion |

### URDF in the ROS 2 Ecosystem

URDF files integrate with key ROS 2 tools:

1. **robot_state_publisher**: Broadcasts transforms between links based on joint states
2. **RViz2**: Visualizes the robot model and sensor data
3. **Gazebo**: Simulates physics and dynamics using URDF-defined properties
4. **MoveIt 2**: Plans collision-free motions using the kinematic model

### Loading URDF in rclpy

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class URDFLoader(Node):
    def __init__(self):
        super().__init__('urdf_loader')

        # Declare and get the robot_description parameter
        self.declare_parameter('robot_description', '')
        robot_description = self.get_parameter('robot_description').value

        if robot_description:
            self.get_logger().info('Robot description loaded successfully')
        else:
            self.get_logger().warn('No robot description found')

def main(args=None):
    rclpy.init(args=args)
    node = URDFLoader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for Humanoid URDF

1. **Use xacro**: For complex humanoids, use xacro macros to reduce repetition
2. **Accurate inertials**: Ensure mass and inertia values match the physical robot
3. **Collision simplification**: Use simpler geometries for collision than visual meshes
4. **Consistent naming**: Follow a systematic naming convention for links and joints
5. **Modular design**: Organize URDF into separate files for each limb/subsystem

---

## Chapter Summary

This chapter established the foundational knowledge required for developing humanoid robots with ROS 2:

- **ROS 2 Architecture**: We explored the core communication patterns—topics for continuous data streams, services for request-response interactions, and actions for long-running tasks with feedback. These patterns enable modular, scalable robot software design.

- **rclpy Programming**: The Python client library provides an accessible entry point to ROS 2 development, allowing creation of nodes, publishers, subscribers, and service clients with clean, Pythonic APIs.

- **URDF Fundamentals**: The Unified Robot Description Format captures the complete physical structure of humanoid robots—from individual links and joints to mass properties and visual representations—enabling seamless integration with simulation, visualization, and motion planning tools.

Together, these foundations prepare you to build sophisticated humanoid robot applications that communicate efficiently, process sensor data, and control complex kinematic structures.

---

## Review Questions

Test your understanding of ROS 2 communication and rclpy concepts:

### 1. Communication Patterns

What are the three main communication patterns in ROS 2, and when would you use each one for a humanoid robot application?

### 2. rclpy Node Lifecycle

Explain the purpose of `rclpy.init()`, `rclpy.spin()`, and `rclpy.shutdown()` in a ROS 2 Python application. Why is proper lifecycle management important?

### 3. Publisher-Subscriber Architecture

A humanoid robot needs to stream joint position data at 100 Hz to a motion controller. Which ROS 2 communication pattern would you use, and why?

### 4. Key Definitions

> **Topic**: A named communication channel over which nodes exchange messages asynchronously using a publish-subscribe model. Topics are the primary mechanism for streaming sensor data, joint states, and command velocities in ROS 2 robotic applications.

> **Node**: The fundamental unit of computation in ROS 2—an executable process that performs a specific task such as reading sensors, processing data, or controlling actuators. Nodes communicate with each other through topics, services, and actions, enabling modular and distributed robot software architectures.
