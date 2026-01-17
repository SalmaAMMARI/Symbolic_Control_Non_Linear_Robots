#!/usr/bin/env python3


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
CONTROL_PERIOD = 0.25          # reduced for tighter control
AUTOMATON_UPDATE_PERIOD = 0.05  # 20 Hz monitoring
MAX_SIM_TIME = 120.0

# Region definitions — identical to symbolic version
R1_REGION = [(4, 5), (8.5, 9.5)]   # '1'
R2_REGION = [(8.5, 9.5), (2, 3)]   # '2'
R3_REGION = [(2, 3), (0.5, 1.5)]   # '3'
R4_REGION = [(3, 7), (3, 7)]       # '4'

REGIONS = {'1': R1_REGION, '2': R2_REGION, '3': R3_REGION, '4': R4_REGION}
CENTERS = {
    '1': np.array([4.5, 9.0]),
    '2': np.array([9.0, 2.5]),
    '3': np.array([2.5, 1.0]),
    '4': np.array([5.0, 5.0])
}

MAX_VX, MAX_VY = 1.5, 1.5
PERTURB_WX = (-0.01, 0.01)
PERTURB_WY = (-0.01, 0.01)

# --------------------------
# UTILITIES
# --------------------------

def point_in_region(pos, region):
    x, y = pos
    (x0, x1), (y0, y1) = region
    return x0 <= x <= x1 and y0 <= y <= y1

def get_continuous_region_label(pos):
    for lbl, reg in REGIONS.items():
        if point_in_region(pos, reg):
            return lbl
    return '0'

def is_position_safe(pos):
    x, y = pos
    if not (0 <= x <= 10 and 0 <= y <= 10):
        return False
    return get_continuous_region_label(pos) == '0'

def sample_random_safe_position(max_trials=1000):
    for _ in range(max_trials):
        pos = np.array([random.uniform(0.5, 9.5), random.uniform(0.5, 9.5)])
        if is_position_safe(pos):
            return pos
    return np.array([1.0, 6.0])  # fallback

# --------------------------
# AUTOMATON
# --------------------------

class AutomatonObjective4:
    def __init__(self):
        self.initial = 's0'
        self.accepting = {'s4'}
        self.transitions = {
            's0': {'1': 's1', '2': 'trap', '3': 'trap', '4': 'trap', '0': 's0'},
            's1': {'1': 's1', '2': 's2', '3': 'trap', '4': 'trap', '0': 's1'},
            's2': {'1': 's2', '2': 's2', '3': 's3', '4': 'trap', '0': 's2'},
            's3': {'1': 's3', '2': 's3', '3': 's3', '4': 's4', '0': 's3'},
            's4': {'1': 's4', '2': 's4', '3': 's4', '4': 's4', '0': 's4'},
            'trap': {'1': 'trap', '2': 'trap', '3': 'trap', '4': 'trap', '0': 'trap'}
        }

    def next_state(self, current, label):
        return self.transitions[current].get(label, 'trap')

    def is_accepting(self, state):
        return state in self.accepting

# --------------------------
# CONTROLLER 
# --------------------------

class PyBulletObjective4Controller:
    def __init__(self, automaton):
        self.automaton = automaton
        self.target_map = {'s0': '1', 's1': '2', 's2': '3', 's3': '4', 's4': '4'}
        self.forbidden = {
            's0': ['2', '3', '4'],
            's1': ['3', '4'],
            's2': ['4'],
            's3': [],
            's4': []
        }

    def get_control(self, x_cont, q):
        candidate_controls = []
        # High-res grid + random sampling
        for vx in np.linspace(-MAX_VX, MAX_VX, 9):
            for vy in np.linspace(-MAX_VY, MAX_VY, 9):
                candidate_controls.append((vx, vy))
        for _ in range(30):
            candidate_controls.append((
                random.uniform(-MAX_VX, MAX_VX),
                random.uniform(-MAX_VY, MAX_VY)
            ))

        best_u = (0.0, 0.0)
        best_score = float('inf')
        target_label = self.target_map.get(q, '4')

        for u in candidate_controls:
            # Predict next position
            x_next = x_cont + np.array(u) * CONTROL_PERIOD
            x_next = np.clip(x_next, [0, 0], [10, 10])

            # REJECT if x_next is in OR NEAR any forbidden region (0.5 unit margin)
            reject = False
            for fr in self.forbidden.get(q, []):
                reg = REGIONS[fr]
                # Expand region by 0.5 units in all directions
                x0, x1 = reg[0][0] - 0.5, reg[0][1] + 0.5
                y0, y1 = reg[1][0] - 0.5, reg[1][1] + 0.5
                if x0 <= x_next[0] <= x1 and y0 <= x_next[1] <= y1:
                    reject = True
                    break
            if reject:
                continue

            # Scoring
            label_next = get_continuous_region_label(x_next)
            dist_to_target = np.linalg.norm(x_next - CENTERS[target_label])

            if label_next == target_label:
                score = dist_to_target - 10.0
            elif label_next in ['0', target_label]:
                score = dist_to_target
            else:
                score = 1e6

            if score < best_score:
                best_score = score
                best_u = u

        return best_u if best_score < 1e9 else None

# --------------------------
# PYBULLET SETUP
# --------------------------

def create_robot(pos):
    v = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.2, 0.6, 1.0, 1])
    c = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
    rid = p.createMultiBody(1.0, c, v, [pos[0], pos[1], 0.3])
    p.changeDynamics(rid, -1, linearDamping=0.5, angularDamping=0.5)
    return rid

def draw_regions():
    colors = {'1': [0.3, 0.3, 1.0], '2': [0.3, 0.8, 0.3], '3': [1.0, 0.8, 0.0], '4': [0.8, 0.3, 0.8]}
    names = {'1': 'R1 (1st)', '2': 'R2 (2nd)', '3': 'R3 (3rd)', '4': 'R4 (4th)'}
    for lbl, reg in REGIONS.items():
        x0, x1 = reg[0]; y0, y1 = reg[1]
        z = 0.01
        pts = [[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z],[x0,y0,z]]
        for i in range(4):
            p.addUserDebugLine(pts[i], pts[i+1], colors[lbl], 3)
        p.addUserDebugText(names[lbl], [(x0+x1)/2, (y0+y1)/2, z+0.1], colors[lbl], 1.5)

def setup_env():
    c = p.connect(p.GUI if USE_GUI else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(TIME_STEP)
    p.loadURDF("plane.urdf")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(12, 45, -45, [5, 5, 0])
    draw_regions()
    return c

def draw_traj(traj, color):
    if len(traj) < 2: return
    for i in range(len(traj)-1):
        p.addUserDebugLine(
            [traj[i][0], traj[i][1], 0.05],
            [traj[i+1][0], traj[i+1][1], 0.05],
            color, 2
        )

# --------------------------
# MAIN
# --------------------------

def main():
    print("="*60)
    print("OBJECTIVE 4 — FIXED: Strict Sequence R1 → R2 → R3 → R4")
    print("="*60)

    automaton = AutomatonObjective4()
    controller = PyBulletObjective4Controller(automaton)

    start_pos = sample_random_safe_position()
    print(f"▶ Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f})")

    setup_env()
    robot_id = create_robot(start_pos)
    p.resetBasePositionAndOrientation(robot_id, [start_pos[0], start_pos[1], 0.3], [0,0,0,1])

    # State
    x_cont = start_pos.copy()
    q = automaton.initial
    sim_time = 0.0
    next_control = 0.0
    next_auto_update = 0.0
    trajectory = [x_cont.copy()]
    visited = set()
    success = False
    trapped = False

    try:
        while sim_time < MAX_SIM_TIME and not success and not trapped:
            # Automaton update (20 Hz)
            if sim_time >= next_auto_update:
                next_auto_update += AUTOMATON_UPDATE_PERIOD
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                x_cont = np.array([pos[0], pos[1]])
                label = get_continuous_region_label(x_cont)

                q_next = automaton.next_state(q, label)
                if q_next != q:
                    print(f" Auto: {q} → {q_next} | label={label} | pos=({x_cont[0]:.2f},{x_cont[1]:.2f})")
                    q = q_next

                    if q == 'trap':
                        print(" TRAP: sequence violation!")
                        trapped = True
                        break

                    if q == 's1' and '1' not in visited:
                        visited.add('1'); print(" R1 visited")
                    elif q == 's2' and '2' not in visited:
                        visited.add('2'); print(" R2 visited")
                    elif q == 's3' and '3' not in visited:
                        visited.add('3'); print(" R3 visited")
                    elif q == 's4':
                        if '4' not in visited:
                            visited.add('4')
                        if label == '4':
                            print("🎉 R4 reached — MISSION COMPLETE!")
                            success = True
                            break

                if automaton.is_accepting(q) and label == '4':
                    success = True
                    break

            # Control update (4 Hz)
            if sim_time >= next_control:
                next_control += CONTROL_PERIOD
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                x_cont = np.array([pos[0], pos[1]])
                trajectory.append(x_cont.copy())

                u = controller.get_control(x_cont, q)
                if u is None:
                    print(" No safe control — halting")
                    p.resetBaseVelocity(robot_id, [0,0,0])
                else:
                    wx = random.uniform(*PERTURB_WX)
                    wy = random.uniform(*PERTURB_WY)
                    vx, vy = u[0] + wx, u[1] + wy
                    p.resetBaseVelocity(robot_id, [vx, vy, 0], [0,0,0])

            p.stepSimulation()
            if USE_GUI:
                time.sleep(TIME_STEP)
            sim_time += TIME_STEP

        # Final report
        print("\n" + "="*50)
        seq = ' → '.join(['R'+lbl for lbl in sorted(visited, key=int)]) if visited else "none"
        if success:
            print(f" SUCCESS at t={sim_time:.1f}s | Sequence: {seq}")
        elif trapped:
            print(f" FAILED at t={sim_time:.1f}s | Violation in region '{get_continuous_region_label(x_cont)}'")
        else:
            print(f" TIMEOUT at t={sim_time:.1f}s | Partial: {seq}")

        draw_traj(trajectory, [0,1,0] if success else [1,0,0] if trapped else [1,0.5,0])

        if USE_GUI:
            print(" Visualization open for 10s...")
            for _ in range(1000):
                p.stepSimulation()
                time.sleep(0.01)

    finally:
        p.disconnect()
        print(" Done")

if __name__ == "__main__":
    main()