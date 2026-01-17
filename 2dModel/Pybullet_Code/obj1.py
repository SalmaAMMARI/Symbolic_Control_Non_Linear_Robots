#!/usr/bin/env python3
"""
Objective 1: Reachability (◊R3) — FIXED
Robust controller with safety margins and continuous validation.
"""

import numpy as np
import random
import time
import pybullet as p
import pybullet_data
import os

# --------------------------
# CONFIGURATION
# --------------------------
USE_GUI = True
TIME_STEP = 1.0 / 240.0
CONTROL_PERIOD = 0.5  # Faster, smoother control
MAX_SIM_TIME = 30.0
TARGET_REGION = [(2, 3), (0.5, 1.5)]  # R3
GOAL_CENTER = np.array([2.5, 1.0])
SAFETY_MARGIN = 0.2  # Avoid overshoot near goal

MAX_VX, MAX_VY = 1.5, 1.5
PERTURB_WX = (-0.01, 0.01)
PERTURB_WY = (-0.01, 0.01)

# --------------------------
# MINIMAL SYMBOLIC ABSTRACTION
# --------------------------

class MinimalRobotAbstraction2D:
    def __init__(self):
        self.state_edges = [np.linspace(0, 10, 50), np.linspace(0, 10, 50)]
        self.index_to_intervals = {}
        n = 49
        for i in range(n):
            for j in range(n):
                idx = i * n + j + 1
                self.index_to_intervals[idx] = [
                    (self.state_edges[0][i], self.state_edges[0][i+1]),
                    (self.state_edges[1][j], self.state_edges[1][j+1])
                ]

    def _find_state(self, x):
        if not (0 <= x[0] <= 10 and 0 <= x[1] <= 10):
            return -1
        i = np.digitize(x[0], self.state_edges[0]) - 1
        j = np.digitize(x[1], self.state_edges[1]) - 1
        n = 49
        return i * n + j + 1 if 0 <= i < n and 0 <= j < n else -1

    def get_center(self, idx):
        if idx == -1 or idx not in self.index_to_intervals:
            return np.array([5, 5])
        iv = self.index_to_intervals[idx]
        return np.array([(iv[0][0] + iv[0][1]) / 2, (iv[1][0] + iv[1][1]) / 2])

# --------------------------
# REACHABILITY CONTROLLER — GREEDY + SAFE
# --------------------------

class ReachabilityController:
    def __init__(self, robot_abs):
        self.robot_abs = robot_abs
        self.controller = {}
        self.Q0 = set()
        self._build_controller()
    
    def _build_controller(self):
        vx_vals = np.linspace(-MAX_VX, MAX_VX, 7)
        vy_vals = np.linspace(-MAX_VY, MAX_VY, 7)
        
        for idx in self.robot_abs.index_to_intervals.keys():
            center = self.robot_abs.get_center(idx)
            best_u = (0.0, 0.0)
            best_score = float('inf')
            safe_controls = []
            
            for vx in vx_vals:
                for vy in vy_vals:
                    next_pos = center + np.array([vx, vy])
                    # Allow all positions (no forbidden regions in Obj 1)
                    if 0 <= next_pos[0] <= 10 and 0 <= next_pos[1] <= 10:
                        dist = np.linalg.norm(next_pos - GOAL_CENTER)
                        score = dist
                        # Bonus for moving toward goal
                        curr_dist = np.linalg.norm(center - GOAL_CENTER)
                        if dist < curr_dist:
                            score -= 2.0
                        safe_controls.append((vx, vy, score))
            
            # Always include stopping
            safe_controls.append((0.0, 0.0, np.linalg.norm(center - GOAL_CENTER)))
            
            if safe_controls:
                # Pick best control by score
                best = min(safe_controls, key=lambda x: x[2])
                self.controller[idx] = best[:2]  # (vx, vy)
                self.Q0.add(idx)
        
        print(f"[Reachability] Controller built with {len(self.controller)} states")

    def get_control(self, state_idx, current_pos):
        if state_idx in self.controller:
            # Adaptive: slow down near goal
            dist_to_goal = np.linalg.norm(current_pos - GOAL_CENTER)
            base_u = self.controller[state_idx]
            if dist_to_goal < 1.0:
                # Reduce speed near goal
                scale = max(0.2, dist_to_goal / 1.0)
                return (base_u[0] * scale, base_u[1] * scale)
            return base_u
        return (0.0, 0.0)  # fallback

# --------------------------
# VISUALIZATION
# --------------------------

def create_robot(pos):
    v = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.2, 0.6, 1.0, 1])
    c = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
    return p.createMultiBody(1.0, c, v, [pos[0], pos[1], 0.3])

def draw_region(region, color, name):
    x0, x1 = region[0]; y0, y1 = region[1]
    z = 0.01
    pts = [[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z],[x0,y0,z]]
    for i in range(4):
        p.addUserDebugLine(pts[i], pts[i+1], color, 3)
    p.addUserDebugText(name, [(x0+x1)/2, (y0+y1)/2, z+0.1], color, 1.5)

def draw_safety_margin(region, color=[1,0.7,0], margin=0.2):
    x0, x1 = region[0][0]-margin, region[0][1]+margin
    y0, y1 = region[1][0]-margin, region[1][1]+margin
    z = 0.005
    pts = [[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z],[x0,y0,z]]
    for i in range(4):
        p.addUserDebugLine(pts[i], pts[i+1], color, 2, lifeTime=10)

def draw_traj(traj, color):
    if len(traj) < 2: return
    for i in range(len(traj)-1):
        p.addUserDebugLine([traj[i][0], traj[i][1], 0.05], [traj[i+1][0], traj[i+1][1], 0.05], color, 2)

def setup_env():
    c = p.connect(p.GUI if USE_GUI else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(TIME_STEP)
    p.loadURDF("plane.urdf")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(12, 45, -45, [5, 5, 0])
    
    draw_region(TARGET_REGION, [1.0, 0.8, 0.0], "R3 GOAL")
    draw_safety_margin(TARGET_REGION, [1, 0.9, 0.5], SAFETY_MARGIN)
    return c

# --------------------------
# MAIN
# --------------------------

def main():
    print("="*60)
    print("OBJECTIVE 1: REACHABILITY (◊R3) — FIXED")
    print("="*60)
    
    robot_abs = MinimalRobotAbstraction2D()
    controller = ReachabilityController(robot_abs)

    if not controller.Q0:
        print(" No initial states")
        return

    # Choose start far from goal
    starts = [s for s in controller.Q0 
              if np.linalg.norm(robot_abs.get_center(s) - GOAL_CENTER) > 3.0]
    s0 = random.choice(starts or list(controller.Q0))
    start_pos = robot_abs.get_center(s0)
    print(f"▶ Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")

    setup_env()
    robot_id = create_robot(start_pos)
    p.resetBasePositionAndOrientation(robot_id, [start_pos[0], start_pos[1], 0.3], [0,0,0,1])

    sim_time = 0.0
    next_control = 0.0
    x_cont = start_pos.copy()
    trajectory = [x_cont.copy()]
    goal_reached = False

    try:
        while sim_time < MAX_SIM_TIME and not goal_reached:
            if sim_time >= next_control:
                next_control += CONTROL_PERIOD
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                x_cont = np.array([pos[0], pos[1]])
                x_sym = robot_abs._find_state(x_cont)

                # Continuous goal check
                in_goal = (TARGET_REGION[0][0] <= x_cont[0] <= TARGET_REGION[0][1] and
                           TARGET_REGION[1][0] <= x_cont[1] <= TARGET_REGION[1][1])
                if in_goal:
                    print(f" GOAL REACHED at t={sim_time:.1f}s!")
                    goal_reached = True
                    break

                # Get control
                u = controller.get_control(x_sym, x_cont)
                wx = random.uniform(*PERTURB_WX)
                wy = random.uniform(*PERTURB_WY)
                p.resetBaseVelocity(robot_id, [u[0]+wx, u[1]+wy, 0], [0,0,0])
                trajectory.append(x_cont.copy())

                dist = np.linalg.norm(x_cont - GOAL_CENTER)
                print(f"t={sim_time:.1f}s | dist_to_goal={dist:.2f} | pos=({x_cont[0]:.2f},{x_cont[1]:.2f})")

            p.stepSimulation()
            if USE_GUI: time.sleep(TIME_STEP)
            sim_time += TIME_STEP

        print("\n" + "="*50)
        if goal_reached:
            print(f" SUCCESS in {sim_time:.1f}s")
        else:
            dist = np.linalg.norm(x_cont - GOAL_CENTER)
            print(f" TIMEOUT | Final distance: {dist:.2f}")

        draw_traj(trajectory, [0,1,0] if goal_reached else [1,0.5,0])

        if USE_GUI:
            for _ in range(500):
                p.stepSimulation()
                time.sleep(0.01)

    finally:
        p.disconnect()
        print(" Done")

if __name__ == "__main__":
    main()