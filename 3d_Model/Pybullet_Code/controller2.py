
import pybullet as p
import pybullet_data
import time
import numpy as np
import itertools
from collections import defaultdict


class RobotAbstraction:
    def __init__(self, state_intervals, control_values, perturbation, delta_t):
        self.state_intervals = state_intervals
        self.control_values = control_values
        self.perturbation = perturbation
        self.delta_t = delta_t
        self.state_to_index = {"OutOfGrid": -1}
        self.index_to_intervals = {}
        self.state_edges = []
        self.discrete_x_y = 20
        self.discrete_tetha = 10
        self.v_vals = 3
        self.omega_vals = 5
        self._create_state_mapping()
        self.compute_transitions_dict()

    def get_successors(self, state_idx, action):
        if not hasattr(self, 'transitions_dict'):
            raise AttributeError("Transitions not computed. Call `robot.compute_transitions_dict()` first.")
        action_key = tuple(np.round(action, decimals=5))
        return list(self.transitions_dict.get((state_idx, action_key), set()))

    def compute_transitions_dict(self):
        transitions = self.compute_transitions()
        self.transitions_dict = {}
        for state, control, successors in transitions:
            control_key = tuple(np.round(control, decimals=5))
            self.transitions_dict[(state, control_key)] = successors

    def _create_state_mapping(self):
        self.state_edges = [
            np.linspace(interval[0], interval[1], self.discrete_x_y) for interval in self.state_intervals[:2]
        ]
        self.state_edges.append(
            np.linspace(self.state_intervals[2][0], self.state_intervals[2][1], self.discrete_tetha)
        )
        state_grid = np.array(np.meshgrid(*[edges[:-1] for edges in self.state_edges])).T.reshape(-1, 3)
        for idx, state in enumerate(state_grid, start=1):
            discrete_state = tuple(
                np.digitize(state[i], self.state_edges[i]) for i in range(len(state))
            )
            self.state_to_index[discrete_state] = idx
            intervals = []
            for i, edge in enumerate(self.state_edges):
                low = edge[discrete_state[i] - 1]
                high = edge[discrete_state[i]]
                intervals.append((low, high))
            self.index_to_intervals[idx] = intervals

    def dynamics(self, x_center, u, w_center, D_x, D_w):
        nominal_dynamics = np.array([
            x_center[0] + self.delta_t * (u[0] * np.cos(x_center[2]) + w_center[0]),
            x_center[1] + self.delta_t * (u[0] * np.sin(x_center[2]) + w_center[1]),
            x_center[2] + self.delta_t * (u[1] + w_center[2])
        ])
        delta_x = np.array([5/99, 5/99, np.pi /30 ])
        delta_w = np.array([0.025, 0.025, 0.025])
        correction = (
            nominal_dynamics - D_x @ delta_x - D_w @ delta_w,
            nominal_dynamics + D_x @ delta_x + D_w @ delta_w
        )
        return correction

    def compute_Dx_and_Dw(self, u):
        D_x = np.array([
            [1, 0, self.delta_t * 0.25],
            [0, 1, self.delta_t * 0.25],
            [0, 0, 1]
        ])
        D_w = np.array([
            [self.delta_t, 0, 0],
            [0, self.delta_t, 0],
            [0, 0, self.delta_t]
        ])
        return D_x, D_w

    def _find_state(self, x):
        discrete_state = tuple(
            min(np.digitize(x[i], self.state_edges[i]), len(self.state_edges[i]) - 1)
            for i in range(len(x))
        )
        return self.state_to_index.get(discrete_state, -1)

    def compute_transitions(self):
        transitions = []
        state_midpoints = [
            np.linspace(interval[0], interval[1], self.discrete_x_y-1) for interval in self.state_intervals[:2]
        ]
        state_midpoints.append(
            np.linspace(self.state_intervals[2][0], self.state_intervals[2][1], self.discrete_tetha-1)
        )
        state_grid = np.array(np.meshgrid(*state_midpoints)).T.reshape(-1, 3)
        v_vals = np.linspace(self.control_values[0][0], self.control_values[0][1], self.v_vals)
        omega_vals = np.linspace(self.control_values[1][0], self.control_values[1][1], self.omega_vals)
        control_grid = np.array(np.meshgrid(v_vals, omega_vals)).T.reshape(-1, 2)
        for i, state in enumerate(state_grid, start=1):
            for control in control_grid:
                successors = set()
                x_center = state
                w_center = np.zeros(3)
                D_x, D_w = self.compute_Dx_and_Dw(control)
                x_next_lower, x_next_upper = self.dynamics(x_center, control, w_center, D_x, D_w)
                possible_states = itertools.product(
                    np.linspace(x_next_lower[0], x_next_upper[0], 3),
                    np.linspace(x_next_lower[1], x_next_upper[1], 3),
                    np.linspace(x_next_lower[2], x_next_upper[2], 3)
                )
                for possible_state in possible_states:
                    state_idx = self._find_state(possible_state)
                    if state_idx != -1:
                        successors.add(state_idx)
                    else:
                        successors.add(-1)
                transitions.append((i, tuple(control), successors))
        return transitions

    def find_indices_for_interval(self, interval):
        overlapping_indices = []
        for index, intervals in self.index_to_intervals.items():
            overlap = True
            for dim in range(len(interval)):
                if not (interval[dim][1] >= intervals[dim][0] and interval[dim][0] <= intervals[dim][1]):
                    overlap = False
                    break
            if overlap:
                overlapping_indices.append(index)
        return overlapping_indices


class SymbolicReachabilityController:
    def __init__(self, robot, target_states):
        self.robot = robot
        self.target_states = target_states
        self.R_star = None
        self.R_sequence = []
        self.H = {}

    def compute_R_star(self):
        print("Computing R* (reachability fixed point)...")
        R_prev = set(self.target_states)
        iteration = 0
        while True:
            iteration += 1
            R_new = set(self.target_states)
            R_new.update(self.pre(R_prev))
            print(f"Iteration {iteration}: |R| = {len(R_new)}")
            self.R_sequence.append(R_new.copy())
            if R_new == R_prev:
                break
            R_prev = R_new
            if iteration > 100:
                print("Warning: Maximum iterations reached")
                break
        self.R_star = R_new
        print(f"R* computation completed: {len(self.R_star)} reachable states")
        return self.R_star

    def pre(self, R):
        pre_states = set()
        for state in range(1, len(self.robot.index_to_intervals) + 1):
            if self._can_reach_set(state, R):
                pre_states.add(state)
        return pre_states

    def _can_reach_set(self, state, target_set):
        for v in [0.5, 1.0, 1.5]:
            for omega in [-1.0, -0.5, 0, 0.5, 1.0]:
                next_states = self._get_next_states(state, (v, omega))
                for next_state in next_states:
                    if next_state in target_set:
                        return True
        return False

    def _get_possible_actions(self, state):
        actions = []
        for v in [0.5, 1.0, 1.5]:
            for omega in [-1.0, -0.5, 0, 0.5, 1.0]:
                actions.append((v, omega))
        return actions

    def _get_next_states(self, state, action):
        next_states = []
        if state not in self.robot.index_to_intervals:
            return next_states
        intervals = self.robot.index_to_intervals[state]
        x_center = (intervals[0][0] + intervals[0][1]) / 2
        y_center = (intervals[1][0] + intervals[1][1]) / 2
        theta_center = (intervals[2][0] + intervals[2][1]) / 2
        v, omega = action
        w_center = np.zeros(3)
        D_x, D_w = self.robot.compute_Dx_and_Dw(action)
        x_next_lower, x_next_upper = self.robot.dynamics(
            np.array([x_center, y_center, theta_center]),
            np.array([v, omega]),
            w_center, D_x, D_w
        )
        sample_points = 3
        for i in range(sample_points):
            for j in range(sample_points):
                for k in range(sample_points):
                    alpha_x = i / (sample_points - 1) if sample_points > 1 else 0.5
                    alpha_y = j / (sample_points - 1) if sample_points > 1 else 0.5
                    alpha_theta = k / (sample_points - 1) if sample_points > 1 else 0.5
                    x_sample = x_next_lower[0] + alpha_x * (x_next_upper[0] - x_next_lower[0])
                    y_sample = x_next_lower[1] + alpha_y * (x_next_upper[1] - x_next_lower[1])
                    theta_sample = x_next_lower[2] + alpha_theta * (x_next_upper[2] - x_next_lower[2])
                    discrete_state = self.robot._find_state(np.array([x_sample, y_sample, theta_sample]))
                    if discrete_state != -1:
                        next_states.append(discrete_state)
        return list(set(next_states))

    def compute_reachability_controller(self):
        if self.R_star is None:
            self.compute_R_star()
        print("Computing reachability controller...")
        for state in self.R_star:
            k_level = None
            for k, R_k in enumerate(self.R_sequence):
                if state in R_k:
                    k_level = k
                    break
            if k_level is not None and k_level > 0:
                target_set = self.R_sequence[k_level - 1]
                valid_actions = []
                for v in [0.5, 1.0, 1.5]:
                    for omega in [-1.0, -0.5, 0, 0.5, 1.0]:
                        next_states = self._get_next_states(state, (v, omega))
                        if any(ns in target_set for ns in next_states):
                            valid_actions.append((v, omega))
                self.H[state] = valid_actions
        print(f"Reachability controller computed for {len(self.H)} states")

    def get_reachability_action(self, continuous_state):
        discrete_state = self.robot._find_state(np.array(continuous_state))
        if discrete_state in self.H and self.H[discrete_state]:
            return self.H[discrete_state][0]
        else:
            print(f"Warning: No reachability action for state {discrete_state}")
            return (1.0, 0.0)


# ==============================================================================
# PYBULLET VISUALIZATION
# ==============================================================================

def main():
    # --- Configuration ---
    state_intervals = [(0, 10), (0, 10), (-np.pi, np.pi)]
    perturbation = [(-0.05, 0.05), (-0.05, 0.05), (-np.pi, np.pi)]
    control_values = [(0.25, 1), (-1, 1)]
    delta_t = 1

    robot = RobotAbstraction(state_intervals, control_values, perturbation, delta_t)

    # Target region R3
    R3_bounds = [[3.0, 4.0], [3.0, 4.0]]
    R3_states = robot.find_indices_for_interval([
        (R3_bounds[0][0], R3_bounds[0][1]),
        (R3_bounds[1][0], R3_bounds[1][1]),
        (robot.state_intervals[2][0], robot.state_intervals[2][1])
    ])

    # Initial state
    initial_state = [1.0, 1.0, 0.0]

    # Controller
    reach_controller = SymbolicReachabilityController(robot, R3_states)
    R_star = reach_controller.compute_R_star()
    reach_controller.compute_reachability_controller()

    # Generate trajectory
    trajectory = [initial_state]
    current = initial_state.copy()
    max_steps = 50
    target_reached = False

    for step in range(max_steps):
        action = reach_controller.get_reachability_action(current)
        w = np.zeros(3)
        D_x, D_w = robot.compute_Dx_and_Dw(action)
        low, up = robot.dynamics(np.array(current), np.array(action), w, D_x, D_w)
        next_state = [(low[i] + up[i]) / 2 for i in range(3)]
        trajectory.append(next_state)
        current = next_state
        if (R3_bounds[0][0] <= current[0] <= R3_bounds[0][1] and
            R3_bounds[1][0] <= current[1] <= R3_bounds[1][1]):
            target_reached = True
            break

    # --- PYBULLET SIMULATION ---
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(
        cameraDistance=8,
        cameraYaw=30,
        cameraPitch=-30,
        cameraTargetPosition=[5, 5, 0]
    )

    # Environment
    p.loadURDF("plane.urdf", [5, 5, 0], globalScaling=10.0)

    # Target region R3 (green cube)
    R3_center_x = (R3_bounds[0][0] + R3_bounds[0][1]) / 2
    R3_center_y = (R3_bounds[1][0] + R3_bounds[1][1]) / 2
    target_vis = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[0.5, 0.5, 0.5],
        rgbaColor=[0, 1, 0, 0.6]
    )
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_vis,
        basePosition=[R3_center_x, R3_center_y, 0.5]
    )

    # Visualize the grid (R* cells as flat green rectangles)
    print("Visualizing R* cells...")
    for state in list(R_star)[:100]:  # Limit to avoid lag
        if state in robot.index_to_intervals:
            x_int, y_int, _ = robot.index_to_intervals[state]
            cx = (x_int[0] + x_int[1]) / 2
            cy = (y_int[0] + y_int[1]) / 2
            sx = x_int[1] - x_int[0]
            sy = y_int[1] - y_int[0]
            visual_shape = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[sx/2, sy/2, 0.01],
                rgbaColor=[0.2, 0.8, 0.2, 0.3]
            )
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape,
                basePosition=[cx, cy, 0.005]
            )

    # Trajectory (line + spheres)
    for i, pt in enumerate(trajectory):
        # Line between points
        if i > 0:
            p.addUserDebugLine(
                [trajectory[i-1][0], trajectory[i-1][1], 0.1],
                [pt[0], pt[1], 0.1],
                [0, 0, 1],  # blue line
                lineWidth=3
            )
        # Sphere at each point
        sphere_vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[0, 0, 1, 0.7]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=sphere_vis,
            basePosition=[pt[0], pt[1], 0.1]
        )

    # Robot
    start_pos = [initial_state[0], initial_state[1], 0.3]
    start_ori = p.getQuaternionFromEuler([0, 0, initial_state[2]])
    robot_id = p.loadURDF("r2d2.urdf", start_pos, start_ori)
    wheel_indices = [2, 3]

    def set_wheel_velocities(v, omega):
        L, r = 0.33, 0.08
        v_l = (v - omega * L / 2) / r
        v_r = (v + omega * L / 2) / r
        p.setJointMotorControl2(robot_id, wheel_indices[0], p.VELOCITY_CONTROL, v_l, force=100)
        p.setJointMotorControl2(robot_id, wheel_indices[1], p.VELOCITY_CONTROL, v_r, force=100)

    # Animate - move robot along trajectory
    print(" Running PyBullet simulation...")
    for i, state in enumerate(trajectory):
        # Target position and orientation
        target_pos = [state[0], state[1], 0.3]
        target_ori = p.getQuaternionFromEuler([0, 0, state[2]])
        
        # Get current position
        if i > 0:
            current_pos, current_ori = p.getBasePositionAndOrientation(robot_id)
        
            for step in range(60):
                alpha = step / 60.0
                # Interpolate position
                interp_pos = [
                    current_pos[0] + alpha * (target_pos[0] - current_pos[0]),
                    current_pos[1] + alpha * (target_pos[1] - current_pos[1]),
                    0.3
                ]
                interp_ori = p.getQuaternionFromEuler([
                    0, 0, 
                    p.getEulerFromQuaternion(current_ori)[2] + 
                    alpha * (state[2] - p.getEulerFromQuaternion(current_ori)[2])
                ])
                
                # Reset robot position
                p.resetBasePositionAndOrientation(robot_id, interp_pos, interp_ori)
                
                # Visual wheel rotation
                v, omega = reach_controller.get_reachability_action(state)
                set_wheel_velocities(v * 10, omega * 10)  # Faster visual rotation
                
                p.stepSimulation()
                time.sleep(1/240)
        else:
            # First state - just pause briefly
            for _ in range(30):
                p.stepSimulation()
                time.sleep(1/240)

    print("Simulation finished.")
    if target_reached:
        print(" SUCCESS: Robot reached the green target zone!")
    else:
        print(" Target not reached within step limit.")

    input("Press Enter to close PyBullet...")
    p.disconnect()


if __name__ == "__main__":
    main()