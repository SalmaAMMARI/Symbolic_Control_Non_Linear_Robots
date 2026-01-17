

import pybullet as p
import pybullet_data
import time
import numpy as np

# ==============================================================================
# GRID ROBOT ABSTRACTION (10x10)
# ==============================================================================

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
        x_pos = min(max(x_pos, 0.0), self.size - 1e-3)
        y_pos = min(max(y_pos, 0.0), self.size - 1e-3)
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
        """Deterministic neighbor for a grid action; clamps at border."""
        if xi == -1 or action not in self.ACTIONS:
            return []
        i, j = self.index_to_ij[xi]
        di, dj = self.DELTAS[action]
        ni, nj = i + di, j + dj
        if 0 <= ni < self.size and 0 <= nj < self.size:
            return [self.ij_to_index[(ni, nj)]]
        return [xi]  # clamp

    def cell_center(self, xi):
        (x0, x1), (y0, y1), _ = self.index_to_intervals[xi]
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    def continuous_step(self, x, action):
        """Small continuous move aligned to action for visualization."""
        x_, y_, theta = x
        dt = 1.0
        speed = 1.0
        dx, dy = 0.0, 0.0
        if action == 'E':
            dx, dy = speed * dt, 0.0; theta = 0.0
        elif action == 'W':
            dx, dy = -speed * dt, 0.0; theta = np.pi
        elif action == 'N':
            dx, dy = 0.0, speed * dt; theta = np.pi / 2
        elif action == 'S':
            dx, dy = 0.0, -speed * dt; theta = -np.pi / 2
        return [x_ + dx, y_ + dy, theta]


# ==============================================================================
# DFA: visit exactly one of R1/R2, avoid R4, then reach R3
# ==============================================================================

class DFA:
    def __init__(self):
        self.states = ['q0', 'q1', 'q2', 'q3', 'q4', 'qfail']
        self.initial_state = 'q0'
        self.accepting_states = {'q3', 'q4'}
        self.transitions = self._build_transitions()

    def _build_transitions(self):
        t = {}
        # q0: choose branch by visiting exactly one of R1/R2; avoid R4
        t[('q0', 'R1')] = 'q1'
        t[('q0', 'R2')] = 'q2'
        t[('q0', 'R3')] = 'q0'
        t[('q0', 'R4')] = 'qfail'
        t[('q0', 'other')] = 'q0'
        # q1: visited R1; visiting R2 -> fail; reaching R3 -> accept (q3)
        t[('q1', 'R1')] = 'q1'
        t[('q1', 'R2')] = 'qfail'
        t[('q1', 'R3')] = 'q3'
        t[('q1', 'R4')] = 'qfail'
        t[('q1', 'other')] = 'q1'
        # q2: visited R2; visiting R1 -> fail; reaching R3 -> accept (q4)
        t[('q2', 'R1')] = 'qfail'
        t[('q2', 'R2')] = 'q2'
        t[('q2', 'R3')] = 'q4'
        t[('q2', 'R4')] = 'qfail'
        t[('q2', 'other')] = 'q2'
        # absorbing
        for sym in ['R1', 'R2', 'R3', 'R4', 'other']:
            t[('q3', sym)] = 'q3'
            t[('q4', sym)] = 'q4'
            t[('qfail', sym)] = 'qfail'
        return t

    def step(self, q, symbol):
        return self.transitions.get((q, symbol), 'qfail')


# ==============================================================================
# Product system (Grid × DFA)
# ==============================================================================

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
        for xi in range(1, self.robot.size * self.robot.size + 1):
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
        """Product successors with safety filter: drop transitions entering qfail."""
        xi, q = self.aug_to_state[aug_idx]
        out = []
        for xn in self.robot.get_successors(xi, action):
            sym = self.label(xn)
            qn = self.dfa.step(q, sym)
            if qn == 'qfail':
                continue
            out.append(self.state_to_aug[(xn, qn)])
        return out


# ==============================================================================
# Backward BFS reachability and controller synthesis (fast)
# ==============================================================================

class DFAReachabilityController:
    def __init__(self, aug):
        self.aug = aug
        self.R_star = set()
        self.level = {}  # BFS level (0 for targets)
        self.H = {}      # multivalued controller: aug_idx -> list of actions

    def actions(self):
        return ['E', 'W', 'N', 'S']

    def compute_R_star(self, max_iters=80):
        """Backward BFS: states that can safely reach accepting targets."""
        print("Computing R* (DFA × Grid) via backward BFS...")
        frontier = set(self.aug.targets)
        self.R_star = set(frontier)
        for t in frontier:
            self.level[t] = 0

        it = 0
        while frontier and it < max_iters:
            it += 1
            new_frontier = set()
            for s in range(len(self.aug.augmented_states)):
                if s in self.R_star:
                    continue
                # include s if exists action with a successor in frontier
                for a in self.actions():
                    succs = self.aug.successors(s, a)
                    if not succs:
                        continue
                    if any(t in frontier for t in succs):
                        self.R_star.add(s)
                        min_succ_level = min([self.level[t] for t in succs if t in self.level], default=it)
                        self.level[s] = min_succ_level + 1
                        new_frontier.add(s)
                        break
            frontier = new_frontier
            print(f"Iteration {it}: |R*| = {len(self.R_star)}")
        print(f"R* size: {len(self.R_star)}")
        return self.R_star

    def compute_controller(self):
        """Pick actions that reduce level and stay safe (same logic as your safe controller)."""
        if not self.R_star:
            self.compute_R_star()
        print("Computing controller policy on R*...")
        for s in self.R_star:
            valid = []
            best_level = float('inf')
            for a in self.actions():
                succs = self.aug.successors(s, a)
                if not succs:
                    continue
                succ_levels = [self.level.get(t, float('inf')) for t in succs]
                min_succ_level = min(succ_levels) if succ_levels else float('inf')
                if min_succ_level < self.level.get(s, float('inf')):
                    if min_succ_level < best_level:
                        valid = [a]; best_level = min_succ_level
                    elif min_succ_level == best_level:
                        valid.append(a)
            if valid:
                self.H[s] = valid
        print(f"Policy computed for {len(self.H)} states.")
        return self.H

    def get_action(self, xi, q):
        """Return first valid action at augmented state, or None if no policy."""
        aug_idx = self.aug.state_to_aug[(xi, q)]
        acts = self.H.get(aug_idx, [])
        return acts[0] if acts else None


# ==============================================================================
# Trajectory generation using the controller (discrete + continuous)
# ==============================================================================

def generate_controller_trajectory(robot, aug, ctrl, initial_state, steps=200):
    """
    Generate the trajectory driven by the synthesized controller (same structure as safe controller):
      - Start from initial continuous position -> discrete cell xi
      - Update DFA on region label each step
      - Use controller action to produce next continuous state with robot.continuous_step
      - Stop when DFA accepts (reach R3 after visiting exactly one of R1/R2) or fail
    Returns: trajectory (continuous points), actions (discrete), spec_ok (bool)
    """
    print("Generating controller-driven trajectory...")
    traj = [initial_state]
    actions = []

    xi = robot._find_state(np.array(initial_state))
    q = aug.dfa.initial_state
    spec_ok = False

    for step in range(steps):
        # Update DFA with current grid label
        sym = aug.label(xi)
        q = aug.dfa.step(q, sym)

        if q == 'qfail':
            print(f" SPEC FAILED at step {step} (entered R4 or visited both R1/R2)")
            break
        if q in aug.dfa.accepting_states:
            spec_ok = True
            print(f" SPEC SATISFIED at step {step} (reached R3 after visiting exactly one of R1/R2)")
            break

        # Controller action; fallback keeps within R* if possible
        a = ctrl.get_action(xi, q)
        if a is None:
            aug_idx = aug.state_to_aug[(xi, q)]
            for cand in ctrl.actions():
                succs = aug.successors(aug_idx, cand)
                if any(s in ctrl.R_star for s in succs):
                    a = cand
                    break
            if a is None:
                a = 'E'  # last resort safe move

        actions.append(a)

        # Apply action (continuous visualization step)
        x_cont_next = robot.continuous_step(traj[-1], a)
        traj.append(x_cont_next)
        # Advance discrete state
        xi = robot.get_successors(xi, a)[0]

        if step % 10 == 0:
            xc, yc = robot.cell_center(xi)
            print(f"Step {step}: cell={xi}, pos≈({xc:.2f},{yc:.2f}), DFA={q}, action={a}")

    return traj, actions, spec_ok


# ==============================================================================
# PYBULLET VISUALIZATION (same logic as your safe reachability visualizer)
# ==============================================================================

def main():
    # --- Configuration ---
    robot = MockRobot(size=10)

    # Regions aligned to grid
    R1_bounds = [[1.0, 2.0], [1.0, 2.0]]   # cell (1,1)
    R2_bounds = [[1.0, 2.0], [7.0, 8.0]]   # cell (1,7)
    R3_bounds = [[7.0, 8.0], [7.0, 8.0]]   # cell (7,7)
    R4_bounds = [[4.0, 6.0], [4.0, 6.0]]   # central 2x2

    # Discrete sets
    R1_states = robot.find_indices_for_interval([R1_bounds[0], R1_bounds[1]])
    R2_states = robot.find_indices_for_interval([R2_bounds[0], R2_bounds[1]])
    R3_states = robot.find_indices_for_interval([R3_bounds[0], R3_bounds[1]])
    R4_states = robot.find_indices_for_interval([R4_bounds[0], R4_bounds[1]])
    print(f"Target R3 states: {len(R3_states)} | Unsafe R4 states: {len(R4_states)}")

    # DFA × Grid product and controller
    dfa = DFA()
    aug = AugmentedSystem(robot, dfa, R1_states, R2_states, R3_states, R4_states)

    ctrl = DFAReachabilityController(aug)
    R_star = ctrl.compute_R_star(max_iters=80)
    ctrl.compute_controller()

    # Initial state
    initial_state = [0.5, 0.5, np.pi/4]

    # Generate trajectory using controller (same pattern as safe controller script)
    trajectory, actions, spec_ok = generate_controller_trajectory(robot, aug, ctrl, initial_state, steps=200)

    # --- PyBullet simulation ---
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(
        cameraDistance=12,
        cameraYaw=45,
        cameraPitch=-35,
        cameraTargetPosition=[robot.size/2, robot.size/2, 0]
    )

    # Environment plane
    p.loadURDF("plane.urdf", [robot.size/2, robot.size/2, 0], globalScaling=robot.size)

    def add_box(bounds, rgba, height=0.5):
        cx = (bounds[0][0] + bounds[0][1]) / 2
        cy = (bounds[1][0] + bounds[1][1]) / 2
        sx = (bounds[0][1] - bounds[0][0]) / 2
        sy = (bounds[1][1] - bounds[1][0]) / 2
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, height/2], rgbaColor=rgba)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[cx, cy, height/2])

    # Regions: visualize like your safe controller
    add_box(R1_bounds, [0.1, 0.4, 1.0, 0.6])  # blue-ish (visit one)
    add_box(R2_bounds, [0.6, 0.2, 0.8, 0.6])  # purple-ish (visit one)
    add_box(R3_bounds, [0.0, 1.0, 0.0, 0.6])  # green (target)
    add_box(R4_bounds, [1.0, 0.0, 0.0, 0.6])  # red (avoid)

    # Visualize R* cells as translucent tiles (limit for performance)
    print("Visualizing R* cells...")
    for aug_idx in list(R_star)[:150]:
        xi, q = aug.aug_to_state[aug_idx]
        if xi in robot.index_to_intervals:
            (x0, x1), (y0, y1), _ = robot.index_to_intervals[xi]
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            sx = (x1 - x0) / 2
            sy = (y1 - y0) / 2
            color = [0.2, 0.8, 0.2, 0.25] if q in dfa.accepting_states else [0.2, 0.6, 0.2, 0.18]
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[sx, sy, 0.01],
                rgbaColor=color
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=[cx, cy, 0.005]
            )

    # Trajectory (lines + spheres)
    traj_color = [0, 0, 1] if spec_ok else [1, 0.5, 0]
    for i, pt in enumerate(trajectory):
        if i > 0:
            p.addUserDebugLine(
                [trajectory[i-1][0], trajectory[i-1][1], 0.1],
                [pt[0], pt[1], 0.1],
                traj_color,
                lineWidth=3
            )
        sphere_vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=traj_color + [0.7]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_vis,
            basePosition=[pt[0], pt[1], 0.1]
        )

    # Robot model and wheel animation (same movement logic)
    start_pos = [initial_state[0], initial_state[1], 0.3]
    start_ori = p.getQuaternionFromEuler([0, 0, initial_state[2]])
    robot_id = p.loadURDF("r2d2.urdf", start_pos, start_ori)
    wheel_indices = [2, 3]

    def set_wheel_velocities_for_action(action):
        # Differential drive params; purely visual spin mapping
        L, r = 0.33, 0.08
        if action == 'E':
            v, omega = 1.0, 0.0
        elif action == 'W':
            v, omega = -1.0, 0.0
        elif action == 'N':
            v, omega = 0.5, 1.5
        else:  # 'S'
            v, omega = -0.5, -1.5
        v_l = (v - omega * L / 2) / r
        v_r = (v + omega * L / 2) / r
        p.setJointMotorControl2(robot_id, wheel_indices[0], p.VELOCITY_CONTROL, v_l, force=100)
        p.setJointMotorControl2(robot_id, wheel_indices[1], p.VELOCITY_CONTROL, v_r, force=100)

    # Animate - move robot along controller-generated trajectory with interpolation
    print("Running PyBullet simulation (DFA × Grid controller)...")
    for i, state in enumerate(trajectory):
        target_pos = [state[0], state[1], 0.3]
        target_ori = p.getQuaternionFromEuler([0, 0, state[2]])
        if i > 0:
            current_pos, current_ori = p.getBasePositionAndOrientation(robot_id)
            action_used = actions[i-1] if (i-1) < len(actions) else 'E'
            for step in range(60):
                alpha = step / 60.0
                interp_pos = [
                    current_pos[0] + alpha * (target_pos[0] - current_pos[0]),
                    current_pos[1] + alpha * (target_pos[1] - current_pos[1]),
                    0.3
                ]
                current_yaw = p.getEulerFromQuaternion(current_ori)[2]
                interp_yaw = current_yaw + alpha * (state[2] - current_yaw)
                interp_ori = p.getQuaternionFromEuler([0, 0, interp_yaw])

                p.resetBasePositionAndOrientation(robot_id, interp_pos, interp_ori)
                set_wheel_velocities_for_action(action_used)

                p.stepSimulation()
                time.sleep(1/240)
        else:
            for _ in range(30):
                p.stepSimulation()
                time.sleep(1/240)

    print("Simulation finished.")
    if spec_ok:
        print(" SUCCESS: Spec satisfied — visited exactly one of R1/R2, then reached R3, avoided R4.")
    else:
        print(" Spec not satisfied within step limit.")

    input("Press Enter to close PyBullet...")
    p.disconnect()


if __name__ == "__main__":
    main()
