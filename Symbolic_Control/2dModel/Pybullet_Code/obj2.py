#!/usr/bin/env python3
"""
Objective 3: ◊(R1∨R2) ∧ ◊R3 ∧ □¬R4 — FULLY FIXED
- Continuous automaton (20 Hz)
- Explicit R4 safety margin in planning
- R3 visit requires dwell time ≥0.2s
- R1/R2 visit only on entry
- No symbolic discretization artifacts
"""

import numpy as np
import random
import time
import pybullet as p
import pybullet_data

# --------------------------
# CONFIGURATION
# --------------------------
USE_GUI = True
TIME_STEP = 1.0 / 240.0
CONTROL_PERIOD = 0.25          # 4 Hz control
AUTOMATON_UPDATE_PERIOD = 0.05  # 20 Hz monitoring
MAX_SIM_TIME = 120.0
SAFETY_MARGIN = 0.3            # R4 avoidance buffer

# Regions — identical to Obj4 for consistency
R1_REGION = [(4, 5), (8.5, 9.5)]   # label '1'
R2_REGION = [(8.5, 9.5), (2, 3)]   # label '2'
R3_REGION = [(2, 3), (0.5, 1.5)]   # label '3'
R4_REGION = [(3, 7), (3, 7)]       # label '4'

REGIONS = {'1': R1_REGION, '2': R2_REGION, '3': R3_REGION, '4': R4_REGION}
CENTERS = {
    '1': np.array([4.5, 9.0]),  # R1
    '2': np.array([9.0, 2.5]),  # R2
    '3': np.array([2.5, 1.0]),  # R3
    'free': np.array([5.0, 5.0])  # fallback
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
    """Returns '1','2','3','4', or '0' (free)"""
    for lbl, reg in REGIONS.items():
        if point_in_region(pos, reg):
            return lbl
    return '0'

def is_position_safe(pos):
    """Strict: no R4 even without margin"""
    return get_continuous_region_label(pos) != '4'

def is_position_robustly_safe(pos):
    """With safety margin: reject if within SAFETY_MARGIN of R4"""
    x, y = pos
    # Expand R4 by SAFETY_MARGIN
    x0, x1 = R4_REGION[0][0] - SAFETY_MARGIN, R4_REGION[0][1] + SAFETY_MARGIN
    y0, y1 = R4_REGION[1][0] - SAFETY_MARGIN, R4_REGION[1][1] + SAFETY_MARGIN
    in_expanded_r4 = x0 <= x <= x1 and y0 <= y <= y1
    in_bounds = 0 <= x <= 10 and 0 <= y <= 10
    return in_bounds and not in_expanded_r4

def sample_random_safe_position(max_trials=1000):
    for _ in range(max_trials):
        pos = np.array([random.uniform(0.5, 9.5), random.uniform(0.5, 9.5)])
        if is_position_robustly_safe(pos):
            return pos
    return np.array([1.0, 6.0])  # fallback (left of R4)

# --------------------------
# AUTOMATON — 
# --------------------------

class AutomatonObjective3:
    def __init__(self):
        self.initial = 'a'
        self.accepting = {'d'}
        self.transitions = {
            'a': {'0': 'a', '1': 'b', '2': 'c', '3': 'a', '4': 'e'},
            'b': {'0': 'b', '1': 'b', '2': 'e', '3': 'd', '4': 'e'},
            'c': {'0': 'c', '1': 'e', '2': 'c', '3': 'd', '4': 'e'},
            'd': {'0': 'd', '1': 'd', '2': 'd', '3': 'd', '4': 'e'},
            'e': {'0': 'e', '1': 'e', '2': 'e', '3': 'e', '4': 'e'}
        }
        # State memory
        self.visited_R1 = False
        self.visited_R2 = False
        self.R3_enter_time = None
        self.R3_dwell = 0.0

    def reset_visit_memory(self):
        self.visited_R1 = False
        self.visited_R2 = False
        self.R3_enter_time = None
        self.R3_dwell = 0.0

    def update_from_label(self, q, label, sim_time):
        # Update visit memory *before* transition
        if label == '1' and not self.visited_R1:
            self.visited_R1 = True
            print(" 🟦 R1 VISITED")
        elif label == '2' and not self.visited_R2:
            self.visited_R2 = True
            print(" 🟩 R2 VISITED")

        # R3 dwell time tracking
        if label == '3':
            if self.R3_enter_time is None:
                self.R3_enter_time = sim_time
            self.R3_dwell = sim_time - self.R3_enter_time
        else:
            self.R3_enter_time = None
            self.R3_dwell = 0.0

        # Transition
        return self.transitions[q].get(label, 'e')

    def is_accepting(self, q):
        if q != 'd':
            return False
        # Require at least one of R1/R2 AND R3 dwell ≥0.2s
        r1_or_r2 = self.visited_R1 or self.visited_R2
        r3_confirmed = self.R3_dwell >= 0.2
        return r1_or_r2 and r3_confirmed

# --------------------------
# CONTROLLER — ROBUST, CONTINUOUS
# --------------------------

class PyBulletObjective3Controller:
    def __init__(self, automaton):
        self.automaton = automaton

    def get_control(self, x_cont, q):
        # Candidate controls: grid + random
        candidates = []
        for vx in np.linspace(-MAX_VX, MAX_VX, 9):
            for vy in np.linspace(-MAX_VY, MAX_VY, 9):
                candidates.append((vx, vy))
        for _ in range(40):
            candidates.append((
                random.uniform(-MAX_VX, MAX_VX),
                random.uniform(-MAX_VY, MAX_VY)
            ))
        candidates.append((0.0, 0.0))  # always allow stop

        # Target region per state
        target_map = {
            'a': ['1', '2'],      # go to R1 or R2
            'b': ['3'],           # go to R3
            'c': ['3'],           # go to R3
            'd': ['3'],           # stay in R3
            'e': []               # trap — no good control
        }
        targets = target_map.get(q, ['free'])
        target_centers = [CENTERS.get(t, CENTERS['free']) for t in targets]

        best_u = (0.0, 0.0)
        best_score = float('inf')

        for u in candidates:
            x_next = x_cont + np.array(u) * CONTROL_PERIOD
            x_next = np.clip(x_next, [0, 0], [10, 10])

            #  REJECT if x_next enters R4 (even with margin)
            if not is_position_robustly_safe(x_next):
                continue

            # Scoring
            label_next = get_continuous_region_label(x_next)
            dist_to_targets = min(np.linalg.norm(x_next - t) for t in target_centers) if target_centers else 1e6

            if label_next in targets:
                score = dist_to_targets - 15.0  # strong incentive
            elif label_next == '0':  # free space
                score = dist_to_targets
            else:
                score = 1e6  # forbidden regions (e.g., wrong service region)

            if score < best_score:
                best_score = score
                best_u = u

        return best_u if best_score < 1e9 else (0.0, 0.0)

# --------------------------
# PYBULLET SETUP
# --------------------------

def create_robot(pos):
    v = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.2, 0.6, 1.0, 1])
    c = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
    rid = p.createMultiBody(1.0, c, v, [pos[0], pos[1], 0.3])
    p.changeDynamics(rid, -1, linearDamping=0.5)
    return rid

def draw_regions():
    colors = {'1': [0.3, 0.3, 1.0], '2': [0.3, 0.8, 0.3], '3': [1.0, 0.8, 0.0], '4': [1.0, 0.3, 0.3]}
    names = {'1': 'R1', '2': 'R2', '3': 'R3 (GOAL)', '4': 'R4 — AVOID!'}
    for lbl, reg in REGIONS.items():
        x0, x1 = reg[0]; y0, y1 = reg[1]
        z = 0.01
        pts = [[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z],[x0,y0,z]]
        for i in range(4):
            p.addUserDebugLine(pts[i], pts[i+1], colors[lbl], 3)
        p.addUserDebugText(names[lbl], [(x0+x1)/2, (y0+y1)/2, z+0.1], colors[lbl], 1.5)
    
    # Draw safety margin around R4
    x0 = R4_REGION[0][0] - SAFETY_MARGIN
    x1 = R4_REGION[0][1] + SAFETY_MARGIN
    y0 = R4_REGION[1][0] - SAFETY_MARGIN
    y1 = R4_REGION[1][1] + SAFETY_MARGIN
    z = 0.005
    pts = [[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z],[x0,y0,z]]
    for i in range(4):
        p.addUserDebugLine(pts[i], pts[i+1], [0.8, 0.2, 0.2], 2, lifeTime=0)

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
        p.addUserDebugLine([traj[i][0], traj[i][1], 0.05], [traj[i+1][0], traj[i+1][1], 0.05], color, 2)

# --------------------------
# MAIN
# --------------------------

def main():
    print("="*60)
    print("OBJECTIVE 3 — FIXED: ◊(R1∨R2) ∧ ◊R3 ∧ □¬R4")
    print("→ Visit R1 *or* R2, then R3, while NEVER entering R4")
    print("="*60)

    automaton = AutomatonObjective3()
    controller = PyBulletObjective3Controller(automaton)

    start_pos = sample_random_safe_position()
    print(f"▶ Start: ({start_pos[0]:.2f}, {start_pos[1]:.2f}) — verified safe")

    setup_env()
    robot_id = create_robot(start_pos)
    p.resetBasePositionAndOrientation(robot_id, [start_pos[0], start_pos[1], 0.3], [0,0,0,1])

    # State
    x_cont = start_pos.copy()
    q = automaton.initial
    automaton.reset_visit_memory()
    sim_time = 0.0
    next_control = 0.0
    next_auto_update = 0.0
    trajectory = [x_cont.copy()]
    success = False
    violation = False

    try:
        while sim_time < MAX_SIM_TIME and not success and not violation:
            #  Automaton update (20 Hz)
            if sim_time >= next_auto_update:
                next_auto_update += AUTOMATON_UPDATE_PERIOD
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                x_cont = np.array([pos[0], pos[1]])
                label = get_continuous_region_label(x_cont)

                # Safety check: exact R4 (hard violation)
                if label == '4':
                    print(f"  SAFETY VIOLATION at t={sim_time:.1f}s (entered R4)")
                    violation = True
                    break

                q_next = automaton.update_from_label(q, label, sim_time)
                if q_next != q:
                    print(f"  Auto: {q} → {q_next} | label={label}")
                    q = q_next
                    if q == 'e':
                        print("  TRAP: sequence violation (e.g., R1→R2 without R3)")
                        violation = True
                        break

                if automaton.is_accepting(q):
                    print(f"  MISSION COMPLETE at t={sim_time:.1f}s!")
                    success = True
                    break

            # 🎮 Control update (4 Hz)
            if sim_time >= next_control:
                next_control += CONTROL_PERIOD
                pos, _ = p.getBasePositionAndOrientation(robot_id)
                x_cont = np.array([pos[0], pos[1]])
                trajectory.append(x_cont.copy())

                u = controller.get_control(x_cont, q)
                wx = random.uniform(*PERTURB_WX)
                wy = random.uniform(*PERTURB_WY)
                vx, vy = u[0] + wx, u[1] + wy
                p.resetBaseVelocity(robot_id, [vx, vy, 0], [0, 0, 0])

                print(f"t={sim_time:.1f}s | Q={q} | pos=({x_cont[0]:.2f},{x_cont[1]:.2f}) | u=({u[0]:.2f},{u[1]:.2f})")

            p.stepSimulation()
            if USE_GUI:
                time.sleep(TIME_STEP)
            sim_time += TIME_STEP

        # Final report
        print("\n" + "="*50)
        if success:
            seq = []
            if automaton.visited_R1: seq.append("R1")
            if automaton.visited_R2: seq.append("R2")
            if automaton.R3_dwell >= 0.2: seq.append("R3")
            print(f" SUCCESS at t={sim_time:.1f}s | Sequence: {' → '.join(seq)}")
        elif violation:
            print(f" FAILED at t={sim_time:.1f}s | Final state: {q}")
        else:
            print(f" TIMEOUT at t={sim_time:.1f}s | Final state: {q}")

        # Visualize
        draw_traj(trajectory, [0, 1, 0] if success else [1, 0, 0] if violation else [1, 0.5, 0])

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