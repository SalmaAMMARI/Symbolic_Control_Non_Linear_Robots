

import numpy as np
import time
import random
from collections import defaultdict
import os
import pybullet as p
import pybullet_data


class MockRobot:
    def __init__(self, size=10):
        self.size = size
        x_edges = np.linspace(0, size, size + 1)
        y_edges = np.linspace(0, size, size + 1)
        theta_edges = np.array([0, 2 * np.pi])
        self.state_edges = [x_edges, y_edges, theta_edges]

        self.index_to_intervals = {}
        self.index_to_ij = {}
        self.ij_to_index = {}
        idx = 1
        for j in range(size):         # row (y)
            for i in range(size):     # col (x)
                self.index_to_intervals[idx] = (
                    (x_edges[i], x_edges[i+1]),
                    (y_edges[j], y_edges[j+1]),
                    (0, 2 * np.pi)
                )
                self.index_to_ij[idx] = (i, j)
                self.ij_to_index[(i, j)] = idx
                idx += 1

    def _find_state(self, x):
        x_pos, y_pos, theta = x
        x_pos = min(max(x_pos, 0.0), self.size - 1e-6)
        y_pos = min(max(y_pos, 0.0), self.size - 1e-6)
        i = int(np.floor(x_pos))
        j = int(np.floor(y_pos))
        return self.ij_to_index.get((i, j), -1)

    def find_indices_for_interval(self, bounds):
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        indices = []
        for idx, ((x0, x1), (y0, y1), _) in self.index_to_intervals.items():
            if x0 < x_max and x1 > x_min and y0 < y_max and y1 > y_min:
                indices.append(idx)
        return indices

    ACTIONS = ['E', 'W', 'N', 'S']
    DELTAS = {
        'E': (1, 0),
        'W': (-1, 0),
        'N': (0, 1),
        'S': (0, -1),
    }

    def get_successors(self, xi, action):
        if xi == -1 or action not in self.ACTIONS:
            return []
        i, j = self.index_to_ij[xi]
        di, dj = self.DELTAS[action]
        ni, nj = i + di, j + dj
        if 0 <= ni < self.size and 0 <= nj < self.size:
            return [self.ij_to_index[(ni, nj)]]
        return [xi]

    def cell_center(self, xi):
        (x0, x1), (y0, y1), _ = self.index_to_intervals[xi]
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def continuous_step(self, x, action):
        x_, y_, theta = x
        dt = 0.95
        speed = 1.0
        dx, dy = 0.0, 0.0
        if action == 'E':
            dx, dy = speed * dt, 0.0
            theta = 0.0
        elif action == 'W':
            dx, dy = -speed * dt, 0.0
            theta = np.pi
        elif action == 'N':
            dx, dy = 0.0, speed * dt
            theta = np.pi / 2
        elif action == 'S':
            dx, dy = 0.0, -speed * dt
            theta = -np.pi / 2
        return [x_ + dx, y_ + dy, theta]


# -------------------------
# DFA for strict sequence R1 → R2 → R3 → R4 → R3
# -------------------------
class DFA:
    def __init__(self):
        self.states = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'qfail']
        self.initial_state = 'q0'
        self.accepting_states = {'q5'}
        self.transitions = self._build()

    def _build(self):
        t = {}
        t[('q0', 'R1')] = 'q1'
        t[('q0', 'other')] = 'q0'
        for r in ['R2', 'R3', 'R4']:
            t[('q0', r)] = 'qfail'
        t[('q1', 'R2')] = 'q2'
        t[('q1', 'R1')] = 'q1'
        t[('q1', 'other')] = 'q1'
        for r in ['R3', 'R4']:
            t[('q1', r)] = 'qfail'
        t[('q2', 'R3')] = 'q3'
        t[('q2', 'R2')] = 'q2'
        t[('q2', 'other')] = 'q2'
        for r in ['R1', 'R4']:
            t[('q2', r)] = 'qfail'
        t[('q3', 'R4')] = 'q4'
        t[('q3', 'R3')] = 'q3'
        t[('q3', 'other')] = 'q3'
        for r in ['R1', 'R2']:
            t[('q3', r)] = 'qfail'
        t[('q4', 'R3')] = 'q5'
        t[('q4', 'R4')] = 'q4'
        t[('q4', 'other')] = 'q4'
        for r in ['R1', 'R2']:
            t[('q4', r)] = 'qfail'
        for sym in ['R1', 'R2', 'R3', 'R4', 'other']:
            t[('q5', sym)] = 'q5'
            t[('qfail', sym)] = 'qfail'
        return t

    def step(self, q, symbol):
        return self.transitions.get((q, symbol), 'qfail')


# -------------------------
# Augmented system (product)
# -------------------------
class AugmentedSystem:
    def __init__(self, robot, dfa, R1_states, R2_states, R3_states, R4_states):
        self.robot = robot
        self.dfa = dfa
        self.R1 = set(R1_states)
        self.R2 = set(R2_states)
        self.R3 = set(R3_states)
        self.R4 = set(R4_states)

        self.augmented_states = []
        self.state_to_aug = {}
        self.aug_to_state = {}
        self.targets = set()
        self._build()

    def _build(self):
        idx = 0
        for xi in range(1, 100 + 1):
            for q in self.dfa.states:
                s = (xi, q)
                self.augmented_states.append(s)
                self.state_to_aug[s] = idx
                self.aug_to_state[idx] = s
                if q in self.dfa.accepting_states:
                    self.targets.add(idx)
                idx += 1

    def label(self, xi):
        if xi in self.R1: return 'R1'
        if xi in self.R2: return 'R2'
        if xi in self.R3: return 'R3'
        if xi in self.R4: return 'R4'
        return 'other'

    def successors(self, aug_idx, action):
        xi, q = self.aug_to_state[aug_idx]
        out = []
        for xn in self.robot.get_successors(xi, action):
            sym = self.label(xn)
            qn = self.dfa.step(q, sym)
            if qn == 'qfail':
                continue
            out.append(self.state_to_aug[(xn, qn)])
        return out


# -------------------------
# Backward reachability and controller synthesis
# -------------------------
class DFAReachabilityController:
    def __init__(self, aug):
        self.aug = aug
        self.R_star = None
        self.layers = []
        self.H = {}

    def actions(self):
        return ['E', 'W', 'N', 'S']

    def pre(self, R):
        A = self.actions()
        preS = set()
        for s in range(len(self.aug.augmented_states)):
            for a in A:
                succs = self.aug.successors(s, a)
                if any(t in R for t in succs):
                    preS.add(s)
                    break
        return preS

    def compute_R_star(self):
        print("Computing R* via backward reachability...")
        R_prev = set(self.aug.targets)
        self.layers = [R_prev.copy()]
        it = 0
        while True:
            it += 1
            R_new = set(self.aug.targets)
            R_new.update(self.pre(R_prev))
            self.layers.append(R_new.copy())
            if R_new == R_prev:
                break
            R_prev = R_new
            if it > 500:
                print("Warning: max iterations reached")
                break
        self.R_star = R_prev
        print(f"R* size: {len(self.R_star)}")
        return self.R_star

    def compute_controller(self):
        if self.R_star is None:
            self.compute_R_star()
        A = self.actions()
        for s in self.R_star:
            k = None
            for i, Ri in enumerate(self.layers):
                if s in Ri:
                    k = i
                    break
            if k is None or k == 0:
                continue
            target_layer = self.layers[k - 1]
            valid = []
            for a in A:
                succs = self.aug.successors(s, a)
                if succs and any(t in target_layer for t in succs):
                    valid.append(a)
            if valid:
                self.H[s] = valid
        print(f"Controller states: {len(self.H)}")
        return self.H


# -------------------------
# Closed-loop simulation
# -------------------------
def run_simulation(robot, aug, ctrl, initial_state, steps=250):
    traj = [initial_state]
    xi = robot._find_state(np.array(initial_state))
    q = aug.dfa.initial_state
    dfa_trace = [q]
    spec_ok = False
    visited = {'R1': False, 'R2': False, 'R3': False, 'R4': False}

    def region_flags(xi_local):
        lab = aug.label(xi_local)
        if lab in visited:
            visited[lab] = True
        return lab

    for step in range(steps):
        lab = region_flags(xi)
        q = aug.dfa.step(q, lab)
        dfa_trace.append(q)
        if q == 'qfail':
            print(f"Specification failed at step {step}")
            break
        if q in aug.dfa.accepting_states:
            spec_ok = True
            print(f"Specification satisfied at step {step}")
            break

        aug_idx = aug.state_to_aug[(xi, q)]

        if aug_idx in ctrl.H and ctrl.H[aug_idx]:
            a = ctrl.H[aug_idx][0]
        else:
            a = None
            for cand in ['E', 'W', 'N', 'S']:
                succs = aug.successors(aug_idx, cand)
                if any(s in ctrl.R_star for s in succs):
                    a = cand
                    break
            if a is None:
                a = random.choice(['E', 'N', 'W', 'S'])

        x_cont = robot.continuous_step(traj[-1], a)
        traj.append(x_cont)
        xi = robot.get_successors(xi, a)[0]

        if step % 10 == 0:
            xc, yc = robot.cell_center(xi)
            print(f"Step {step}: cell={xi}, pos≈({xc:.2f},{yc:.2f}), DFA={q}, action={a}")

    return traj, dfa_trace, spec_ok, visited


# -------------------------
# PyBullet visualization (improved)
# -------------------------
def run_pybullet_visualization(robot, trajectory, R1_bounds, R2_bounds, R3_bounds, R4_bounds, pause_between_steps=0.06):
    # Ensure no leftover connection
    try:
        if p.isConnected():
            p.disconnect()
    except Exception:
        pass

    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    # Hide GUI extras
    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    except Exception:
        pass

    # Dark ground
    ground_half = max(robot.size, 10) / 2.0
    ground_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[ground_half, ground_half, 0.01],
                                     rgbaColor=[0.08, 0.08, 0.08, 1.0])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=ground_vis, basePosition=[robot.size/2, robot.size/2, -0.01])

    # Top-down camera
    p.resetDebugVisualizerCamera(cameraDistance=14, cameraYaw=45, cameraPitch=-60,
                                 cameraTargetPosition=[robot.size/2, robot.size/2, 0])

    # Grid lines
    grid_color = [0.25, 0.25, 0.25]
    for x_edge in robot.state_edges[0]:
        p.addUserDebugLine([x_edge, robot.state_edges[1][0], 0.01],
                           [x_edge, robot.state_edges[1][-1], 0.01],
                           lineColorRGB=grid_color, lineWidth=1.0)
    for y_edge in robot.state_edges[1]:
        p.addUserDebugLine([robot.state_edges[0][0], y_edge, 0.01],
                           [robot.state_edges[0][-1], y_edge, 0.01],
                           lineColorRGB=grid_color, lineWidth=1.0)

    # Regions
    def add_region(bounds, rgba, height=0.02):
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        hx = 0.5 * (x_max - x_min)
        hy = 0.5 * (y_max - y_min)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, height], rgbaColor=rgba)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[cx, cy, height/2])

    add_region(R1_bounds, [0.15, 0.35, 0.95, 0.45])
    add_region(R2_bounds, [0.6, 0.2, 0.8, 0.45])
    add_region(R3_bounds, [0.15, 0.95, 0.35, 0.45])
    add_region(R4_bounds, [0.95, 0.15, 0.15, 0.45])

    # Trajectory polyline and markers
    traj_color = [0.0, 0.8, 0.2]
    for i in range(len(trajectory)):
        if i > 0:
            p.addUserDebugLine([trajectory[i-1][0], trajectory[i-1][1], 0.12],
                               [trajectory[i][0], trajectory[i][1], 0.12],
                               traj_color, lineWidth=3)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=traj_color + [0.9])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[trajectory[i][0], trajectory[i][1], 0.12])

    # Robot
    start_pos = [trajectory[0][0], trajectory[0][1], 0.25]
    start_ori = p.getQuaternionFromEuler([0, 0, trajectory[0][2]])
    robot_id = None
    try:
        robot_id = p.loadURDF("r2d2.urdf", start_pos, start_ori, globalScaling=0.8)
        print("Using r2d2.urdf")
    except Exception as e:
        print("r2d2.urdf not found or failed, using sphere fallback:", e)
        sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.12)
        sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.12, rgbaColor=[0.0, 0.8, 0.5, 1.0])
        robot_id = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=sphere_col,
                                     baseVisualShapeIndex=sphere_vis, basePosition=start_pos)

    # ✅ Force initial pose (critical fix!)
    p.resetBasePositionAndOrientation(robot_id, start_pos, start_ori)

    def draw_orientation_arrow(pos, yaw, length=0.35, color=[1, 1, 0]):
        start = [pos[0], pos[1], 0.25]
        end = [pos[0] + length * np.cos(yaw), pos[1] + length * np.sin(yaw), 0.25]
        p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=3.0, lifeTime=0.04)

    print("Animating robot in PyBullet...")
    try:
        for i, state in enumerate(trajectory):
            target_pos = [state[0], state[1], 0.25]
            target_yaw = state[2]

            if i == 0:
                # Initial pose already set; just show it
                for _ in range(30):
                    draw_orientation_arrow(target_pos, target_yaw)
                    p.addUserDebugText(f"{i}", [target_pos[0], target_pos[1], 0.6], textColorRGB=[1,1,1], textSize=1.0, lifeTime=0.04)
                    p.stepSimulation()
                    time.sleep(1/240.0)
                continue

            # Get current pose
            current_pos, current_ori = p.getBasePositionAndOrientation(robot_id)
            current_yaw = p.getEulerFromQuaternion(current_ori)[2]

            # Normalize angle difference
            dyaw = (target_yaw - current_yaw + np.pi) % (2*np.pi) - np.pi
            interp_steps = max(15, int(0.1 * 240))  # ~0.1 sec per segment

            for step in range(interp_steps):
                alpha = (step + 1) / interp_steps
                interp_pos = [
                    current_pos[0] + alpha * (target_pos[0] - current_pos[0]),
                    current_pos[1] + alpha * (target_pos[1] - current_pos[1]),
                    0.25
                ]
                interp_yaw = current_yaw + alpha * dyaw
                interp_ori = p.getQuaternionFromEuler([0, 0, interp_yaw])
                p.resetBasePositionAndOrientation(robot_id, interp_pos, interp_ori)

                draw_orientation_arrow(interp_pos, interp_yaw)
                p.addUserDebugText(f"{i}", [interp_pos[0], interp_pos[1], 0.6], textColorRGB=[1,1,1], textSize=1.0, lifeTime=0.04)
                p.stepSimulation()
                time.sleep(1/240.0)

    except KeyboardInterrupt:
        print("Animation interrupted by user.")
    finally:
        print("Keeping PyBullet window open. Close it manually to exit.")
        try:
            while p.isConnected():
                time.sleep(0.1)
        except:
            pass
        finally:
            p.disconnect()


# -------------------------
# Main
# -------------------------
def main():
    R1_bounds = [[1.0, 2.0], [1.0, 2.0]]
    R2_bounds = [[1.0, 2.0], [7.0, 8.0]]
    R3_bounds = [[7.0, 8.0], [7.0, 8.0]]
    R4_bounds = [[4.0, 6.0], [4.0, 6.0]]

    robot = MockRobot()
    R1_states = robot.find_indices_for_interval(R1_bounds)
    R2_states = robot.find_indices_for_interval(R2_bounds)
    R3_states = robot.find_indices_for_interval(R3_bounds)
    R4_states = robot.find_indices_for_interval(R4_bounds)

    print(f"R1: {len(R1_states)}, R2: {len(R2_states)}, R3: {len(R3_states)}, R4: {len(R4_states)}")

    dfa = DFA()
    aug = AugmentedSystem(robot, dfa, R1_states, R2_states, R3_states, R4_states)
    ctrl = DFAReachabilityController(aug)
    ctrl.compute_R_star()
    ctrl.compute_controller()

    initial_state = [0.5, 0.5, np.pi/4]
    trajectory, dfa_trace, ok, visited = run_simulation(robot, aug, ctrl, initial_state, steps=250)
    print(f"Trajectory length: {len(trajectory)}, spec satisfied: {ok}")

    run_pybullet_visualization(robot, trajectory, R1_bounds, R2_bounds, R3_bounds, R4_bounds)


if __name__ == "__main__":
    main()