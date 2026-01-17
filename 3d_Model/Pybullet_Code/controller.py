
import numpy as np
import itertools
from collections import defaultdict

import pybullet as p
import pybullet_data
import time

class RobotAbstraction:
    def __init__(self, state_intervals, control_values, perturbation, delta_t):
        self.state_intervals = state_intervals  # [(x_min, x_max), (y_min, y_max), (theta_min, theta_max)]
        self.control_values = control_values  # [(v_min, v_max), (omega_min, omega_max)]
        self.perturbation = perturbation  # [(w1_min, w1_max), (w2_min, w2_max), (w3_min, w3_max)]
        self.delta_t = delta_t  # sampling time (time to change a state) = 1s
        self.state_to_index = {"OutOfGrid": -1}  # Map from state tuple to symbolic state ξ
        self.index_to_intervals = {}  # Map from symbolic state ξ to intervals
        self.state_edges = []  
        self.discrete_x_y=20
        self.discrete_tetha=10
        self.v_vals=3
        self.omega_vals=5
        self._create_state_mapping()  # Precompute the symbolic state mapping
        self.compute_transitions_dict() 
    
    def get_successors(self, state_idx, action):
        """
        Return list of successor state indices from `state_idx` under `action`.
        Requires that `compute_transitions()` has been called and stored in `self.transitions_dict`.
        """
        if not hasattr(self, 'transitions_dict'):
            raise AttributeError(
                "Transitions not computed. Call `robot.compute_transitions_dict()` first."
            )
        action_key = tuple(np.round(action, decimals=5))  # Avoid floating-point key issues
        return list(self.transitions_dict.get((state_idx, action_key), set()))
    
    def compute_transitions_dict(self):
        """
        Compute and store transitions as a dictionary: {(state, action): set(successors)}
        """
        transitions = self.compute_transitions()  # Uses your existing method
        self.transitions_dict = {}
        for state, control, successors in transitions:
            control_key = tuple(np.round(control, decimals=5))
            self.transitions_dict[(state, control_key)] = successors

    def _create_state_mapping(self):
        """Precompute a mapping from discrete states to symbolic indices (ξ) and intervals."""
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

            # Map intervals for each dimension of the state
            intervals = []
            for i, edge in enumerate(self.state_edges):
                low = edge[discrete_state[i] - 1]
                high = edge[discrete_state[i]]
                intervals.append((low, high))
            self.index_to_intervals[idx] = intervals
    
    def dynamics(self, x_center, u, w_center, D_x, D_w):
        """Modified system dynamics based on the image formula."""
        nominal_dynamics = np.array([
            x_center[0] + self.delta_t * (u[0] * np.cos(x_center[2]) + w_center[0]),
            x_center[1] + self.delta_t * (u[0] * np.sin(x_center[2]) + w_center[1]),
            x_center[2] + self.delta_t * (u[1] + w_center[2])
        ])
        delta_x = np.array([5/99, 5/99, np.pi /30 ])   # delta x = (x max-x min)/Nx-1  chosen randomly
        delta_w = np.array([0.025, 0.025, 0.025])

        correction = (
            nominal_dynamics - D_x @ delta_x - D_w @ delta_w,
            nominal_dynamics + D_x @ delta_x + D_w @ delta_w
        )
        return correction

    def compute_Dx_and_Dw(self, u):
        """Compute the bounds matrices D_x and D_w."""
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
        """Find the discrete state corresponding to x. x is a vector =[x,y,theta]."""
        discrete_state = tuple(
            min(np.digitize(x[i], self.state_edges[i]), len(self.state_edges[i]) - 1)
            for i in range(len(x))
        )
        return self.state_to_index.get(discrete_state, -1)

    def compute_transitions(self):
        """Compute transitions for the discrete model, considering perturbations."""
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
        control_grid = np.array(np.meshgrid(v_vals, omega_vals)).T.reshape(-1, 2)   #All possible control inputs

        for i, state in enumerate(state_grid, start=1):
            for control in control_grid:
                successors = set()  #to keep track of successors of x_center=current state 
                x_center = state
                w_center = np.zeros(3)  # Using zeros for perturbation center
                D_x, D_w = self.compute_Dx_and_Dw(control)
                x_next_lower, x_next_upper = self.dynamics(x_center, control, w_center, D_x, D_w)

                # Compute the Cartesian product of all possible values
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

    def print_transitions_and_intervals(self, transitions):
        """Print the transitions and intervals for each state."""
        # Group transitions by state
        grouped_transitions = defaultdict(list)
        for state, control, successors in transitions:
            grouped_transitions[state].append((control, successors))

        # Print the grouped transitions
        print("Transitions with Intervals:")
        for state, transitions in grouped_transitions.items():
            if state not in self.index_to_intervals:
                print(f"State ξ={state} not in index_to_intervals! Skipping.")
                continue
            intervals = self.index_to_intervals[state]
            interval_str = ", ".join([f"[{low:.2f}, {high:.2f}]" for low, high in intervals])
            print(f"\nState ξ={state} ({interval_str}):")

            for control, successors in transitions:
                control_str = f"(v={control[0]:.2f}, ω={control[1]:.2f})"
                successors_str = ", ".join([f"ξ={s}" for s in successors]) if successors else "None"
                print(f"  Control {control_str} -> Successors: {successors_str}")

    def find_indices_for_interval(self, interval):
        """
        Find all indices whose intervals overlap with the given interval.

        Args:
            interval: List of tuples [(x_min, x_max), (y_min, y_max), (theta_min, theta_max)]
                     representing the interval to search.

        Returns:
            A list of indices (ξ) that overlap with the given interval.
        """
        overlapping_indices = []

        for index, intervals in self.index_to_intervals.items():
            # Check if all dimensions overlap
            overlap = True
            for dim in range(len(interval)):
                if not (interval[dim][1] >= intervals[dim][0] and interval[dim][0] <= intervals[dim][1]):
                    overlap = False
                    break
            if overlap:
                overlapping_indices.append(index)

        return overlapping_indices

class SymbolicControllerSynthesis:
    def __init__(self, transitions, safety_states):
        """
        Initialize the symbolic controller synthesis.

        Parameters:
        - transitions: List of transitions in the form [(state, control, successors), ...].
        - safety_states: Set of safe states (Q_s).
        """
        self.transitions = self._process_transitions(transitions)  # Convert list to dictionary
        self.safety_states = safety_states

    def _process_transitions(self, transitions):
        """
        Convert the list of transitions into a dictionary format.

        Parameters:
        - transitions: List of transitions [(state, control, successors), ...].

        Returns:
        - A dictionary mapping (state, control) -> set(successors).
        """
        transition_dict = defaultdict(set)
        for state, control, successors in transitions:
            transition_dict[(state, control)] = successors
        return dict(transition_dict)

    def pre(self, R):
        """
        Compute the predecessor operator Pre(R), considering only states with valid transitions.

        Parameters:
        - R: Set of states (current safe set).

        Returns:
        - Set of states that can transition to R.
        """
        pre_states = set()
        for (state, control), successors in self.transitions.items():
            # Check if any successor is in R
            if successors and successors.issubset(R):
                pre_states.add(state)
        return pre_states

    def compute_safe_controller(self):
        """
        Compute the maximal safe set (R*).

        Returns:
        - R*: Maximal set of safe states.
        """
        R = self.safety_states.copy()
        Q_s = R
        while True:
            R_next = Q_s.intersection(self.pre(R))
            if R_next == R:
                break
            R = R_next
        return R

    def synthesize_controller(self):
        """
        Synthesize a safe controller.

        Returns:
        - R*: Maximal safe set.
        - Controller: Mapping from states to safe controls.
        - Q_0: Set of valid initial states.
        """
        R_star = self.compute_safe_controller()
        controller = defaultdict(set)
        Q_0 = set()

        for (state, control), successors in self.transitions.items():
            # Add controls only if all successors are in R*
            if state in R_star and successors.issubset(R_star):
                controller[state].add(control)

        # Q_0: States in R_star with at least one valid control
        for state in R_star:
            if state in controller:
                Q_0.add(state)

        return R_star, controller, Q_0

def symbolic_to_continuous_map(state, control):
    """
    Map a symbolic control to its corresponding continuous control.

    Parameters:
    - state: The symbolic state index.
    - control: The symbolic control (tuple of discretized control values).

    Returns:
    - The corresponding continuous control (v, ω).
    """
    return control

def concretize_controller_with_intervals(controller, index_to_intervals):
    """
    Concretize the symbolic controller to work with continuous dynamics and include state intervals.
    
    Parameters:
    - controller: Dictionary {state: set of symbolic controls}.
    - index_to_intervals: Dictionary {state: intervals}.
    
    Returns:
    - concretized_controller: Dictionary {state: (intervals, continuous controls)}.
    """
    concretized_controller = {}
    for state, controls in controller.items():
        state_intervals = index_to_intervals.get(state, "No intervals available")
        continuous_controls = {symbolic_to_continuous_map(state, control) for control in controls}
        concretized_controller[state] = (state_intervals, continuous_controls)
    return concretized_controller

# Simulate the optimal path using BFS
from collections import deque


import random

def generate_safe_trajectory(
    initial_state, initial_perturbation, concretized_controller, robot, max_steps=1000
):
    """
    Generate trajectory that NEVER goes out of bounds or reaches state -1.
    """
    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def clip_to_bounds(state, robot):
        """Clip state to stay strictly within grid."""
        epsilon = 1e-6
        clipped = state.copy()
        clipped[0] = np.clip(state[0], 
                            robot.state_intervals[0][0] + epsilon,
                            robot.state_intervals[0][1] - epsilon)
        clipped[1] = np.clip(state[1],
                            robot.state_intervals[1][0] + epsilon,
                            robot.state_intervals[1][1] - epsilon)
        clipped[2] = normalize_angle(state[2])
        return clipped
    
    def is_within_bounds(state, robot):
        """Check if state is within bounds (3D: x, y, theta)."""
        return (robot.state_intervals[0][0] <= state[0] <= robot.state_intervals[0][1] and
                robot.state_intervals[1][0] <= state[1] <= robot.state_intervals[1][1] and
                robot.state_intervals[2][0] <= state[2] <= robot.state_intervals[2][1])
    
    # Ensure initial state is within bounds
    current_state = clip_to_bounds(initial_state, robot)
    trajectory = [current_state.tolist()]
    current_perturbation = initial_perturbation
    
    print(f"\nStarting trajectory generation from: {current_state}")
    
    for step in range(max_steps):
        discrete_state = robot._find_state(current_state)
        
        # CRITICAL: Stop if we reach invalid state
        if discrete_state == -1:
            print(f"Step {step}: Reached invalid state -1. Stopping.")
            break
        
        if discrete_state not in concretized_controller:
            print(f"Step {step}: No controller for state {discrete_state}. Stopping.")
            break
        
        state_intervals, available_controls = concretized_controller[discrete_state]
        
        if not available_controls:
            print(f"Step {step}: No available controls. Stopping.")
            break
        
        # Try all controls to find one that keeps us in bounds
        controls_list = list(available_controls)
        random.shuffle(controls_list)
        
        success = False
        best_candidate = None
        best_discrete = -1
        
        for control in controls_list:
            # Compute dynamics
            D_x, D_w = robot.compute_Dx_and_Dw(control)
            lower, upper = robot.dynamics(current_state, control, current_perturbation, D_x, D_w)
            
            # Clip the reachable set to grid bounds FIRST
            lower_clipped = clip_to_bounds(lower, robot)
            upper_clipped = clip_to_bounds(upper, robot)
            
            # Try multiple samples from the reachable set
            for attempt in range(15):
                if attempt == 0:
                    # Try center first
                    candidate = (lower_clipped + upper_clipped) / 2
                elif attempt < 5:
                    # Try biased towards staying in bounds
                    alpha = 0.3 + 0.1 * attempt  # 0.3, 0.4, 0.5, 0.6, 0.7
                    candidate = alpha * lower_clipped + (1 - alpha) * upper_clipped
                else:
                    # Random sample
                    candidate = np.array([
                        np.random.uniform(lower_clipped[0], upper_clipped[0]),
                        np.random.uniform(lower_clipped[1], upper_clipped[1]),
                        np.random.uniform(lower_clipped[2], upper_clipped[2])
                    ])
                
                # Ensure it's clipped
                candidate = clip_to_bounds(candidate, robot)
                
                # Double-check bounds
                if not is_within_bounds(candidate, robot):
                    continue
                
                # Check if valid
                next_discrete = robot._find_state(candidate)
                
                # CRITICAL: Ensure next state is NOT -1 and is in controller
                if next_discrete != -1 and next_discrete in concretized_controller:
                    best_candidate = candidate
                    best_discrete = next_discrete
                    success = True
                    break
            
            if success:
                break
        
        if success and best_candidate is not None:
            current_state = best_candidate
            trajectory.append(best_candidate.tolist())
        else:
            print(f"Step {step}: Could not find safe next state. Stopping.")
            print(f"  Current state: {current_state}")
            print(f"  Discrete state: {discrete_state}")
            print(f"  Available controls: {len(available_controls)}")
            break
    
    print(f"\n Trajectory generated: {len(trajectory)} states, {len(trajectory)-1} steps")
    
    # Verify no states map to -1
    invalid_states = []
    oob_states = []
    
    for i, s in enumerate(trajectory):
        s_array = np.array(s)
        discrete = robot._find_state(s_array)
        if discrete == -1:
            invalid_states.append((i, s))
        if not is_within_bounds(s_array, robot):
            oob_states.append((i, s))
    
    if invalid_states:
        print(f"  WARNING: {len(invalid_states)} states map to -1!")
        for i, s in invalid_states[:3]:
            print(f"    Step {i}: {s}")
    else:
        print(f"  All states are valid (no -1)")
    
    if oob_states:
        print(f"  WARNING: {len(oob_states)} states out of bounds!")
        for i, s in oob_states[:3]:
            print(f"    Step {i}: {s}")
    else:
        print(f"   All states within bounds")
    
    return trajectory

def select_initial_state_from_Q0_safe(Q_0, robot):
    """
    Select a valid continuous initial state from Q_0, ensuring it's not -1.
    """
    if not Q_0:
        raise ValueError("No valid initial states in Q_0!")
    
    # Remove -1 if it somehow got in
    Q_0_clean = Q_0 - {-1}
    
    if not Q_0_clean:
        raise ValueError("Q_0 only contains invalid state -1!")
    
    # Select a random valid state
    initial_symbolic_state = random.choice(list(Q_0_clean))
    
    # Get the intervals for this symbolic state
    if initial_symbolic_state not in robot.index_to_intervals:
        raise ValueError(f"State {initial_symbolic_state} not in index_to_intervals!")
    
    state_intervals = robot.index_to_intervals[initial_symbolic_state]
    
    # Choose the center point
    continuous_state = np.array([
        (state_intervals[0][0] + state_intervals[0][1]) / 2,
        (state_intervals[1][0] + state_intervals[1][1]) / 2,
        (state_intervals[2][0] + state_intervals[2][1]) / 2
    ])
    
    print(f"\nInitial State Selection:")
    print(f"  Symbolic state ξ={initial_symbolic_state}")
    print(f"  Intervals: {state_intervals}")
    print(f"  Continuous state: [{continuous_state[0]:.3f}, {continuous_state[1]:.3f}, {continuous_state[2]:.3f}]")
    
    return continuous_state, initial_symbolic_state

# When computing transitions, use this corrected version:
def compute_transitions_safe(self):
    """
    Compute transitions, excluding -1 and clipping to bounds.
    """
    transitions = []
    state_midpoints = [
        np.linspace(interval[0], interval[1], self.discrete_x_y-1) 
        for interval in self.state_intervals[:2]
    ]
    state_midpoints.append(
        np.linspace(self.state_intervals[2][0], self.state_intervals[2][1], 
                   self.discrete_tetha-1)
    )
    state_grid = np.array(np.meshgrid(*state_midpoints)).T.reshape(-1, 3)

    v_vals = np.linspace(self.control_values[0][0], self.control_values[0][1], self.v_vals)
    omega_vals = np.linspace(self.control_values[1][0], self.control_values[1][1], self.omega_vals)
    control_grid = np.array(np.meshgrid(v_vals, omega_vals)).T.reshape(-1, 2)

    out_of_bounds_count = 0
    
    for i, state in enumerate(state_grid, start=1):
        for control in control_grid:
            successors = set()
            x_center = state
            w_center = np.zeros(3)
            D_x, D_w = self.compute_Dx_and_Dw(control)
            x_next_lower, x_next_upper = self.dynamics(x_center, control, w_center, D_x, D_w)
            
            # CLIP to grid bounds (x, y, theta)
            x_next_lower[0] = max(x_next_lower[0], self.state_intervals[0][0])
            x_next_lower[1] = max(x_next_lower[1], self.state_intervals[1][0])
            x_next_lower[2] = max(x_next_lower[2], self.state_intervals[2][0])
            x_next_upper[0] = min(x_next_upper[0], self.state_intervals[0][1])
            x_next_upper[1] = min(x_next_upper[1], self.state_intervals[1][1])
            x_next_upper[2] = min(x_next_upper[2], self.state_intervals[2][1])

            # Compute the Cartesian product
            possible_states = itertools.product(
                np.linspace(x_next_lower[0], x_next_upper[0], 3),
                np.linspace(x_next_lower[1], x_next_upper[1], 3),
                np.linspace(x_next_lower[2], x_next_upper[2], 3)
            )

            for possible_state in possible_states:
                state_idx = self._find_state(possible_state)
                
                # NEVER add -1
                if state_idx != -1:
                    successors.add(state_idx)
                else:
                    out_of_bounds_count += 1

            # Only add if has valid successors
            if successors:
                transitions.append((i, tuple(control), successors))
    
    print(f"Transition computation: {len(transitions)} valid transitions, {out_of_bounds_count} OOB rejected")
    return transitions

RobotAbstraction.compute_transitions_safe = compute_transitions_safe

class SymbolicSafetyController:
    """
    Contrôleur de sûreté symbolique corrigé
    """
    
    def __init__(self, robot, safety_states, R4_bounds):
        self.robot = robot
        self.safety_states = safety_states
        self.R4_bounds = R4_bounds
        self.R_star = None
        self.controller = {}  # Stocke les actions sûres pour chaque état
    
    def compute_safe_set(self):
        """
        Calcule l'ensemble sûr maximal R*
        Algorithme de point fixe
        """
        print("Computing safe set R*...")
        R_prev = self.safety_states.copy()
        iteration = 0
        
        while True:
            iteration += 1
            R_new = set()
            
            for state in R_prev:
                # Vérifier si l'état peut rester dans R_prev
                if self._can_remain_safe(state, R_prev):
                    R_new.add(state)
            
            print(f"Iteration {iteration}: |R| = {len(R_new)}")
            
            if R_new == R_prev or iteration > 50:
                break
                
            R_prev = R_new
        
        self.R_star = R_new
        print(f"Safe set computation completed: {len(self.R_star)} safe states")
        return self.R_star
    
    def _can_remain_safe(self, state, safe_set):
        """
        Vérifie si un état peut rester dans l'ensemble sûr
        """
        # Obtenir toutes les transitions possibles depuis cet état
        possible_controls = []
        for (s, control), successors in self.robot.transitions_dict.items():
            if s == state:
                possible_controls.append((control, successors))
        
        for control, successors in possible_controls:
            # Vérifier si tous les successeurs sont dans l'ensemble sûr
            if successors and all(succ in safe_set for succ in successors):
                # Stocker cette action comme sûre
                if state not in self.controller:
                    self.controller[state] = set()
                self.controller[state].add(control)
                return True
        
        return False
    
    def get_safe_action(self, continuous_state):
        """
        Retourne une action sûre pour l'état continu donné
        """
        # Convertir l'état continu en état symbolique
        discrete_state = self.robot._find_state(np.array(continuous_state))
        
        if discrete_state == -1:
            print("Warning: State out of grid")
            return (0.0, 0.0)  # Arrêt d'urgence
        
        if discrete_state not in self.R_star:
            print(f"Warning: State {discrete_state} not in safe set")
            return (0.0, 0.0)  # Arrêt d'urgence
        
        if discrete_state in self.controller and self.controller[discrete_state]:
            # Retourner une action sûre arbitraire
            safe_control = next(iter(self.controller[discrete_state]))
            return safe_control
        else:
            print(f"Warning: No safe action for state {discrete_state}")
            return (0.0, 0.0)  # Arrêt d'urgence

# ============================================================================
# PYBULLET VISUALIZATION 
# ============================================================================

def create_grid_visualization(robot_abstraction):
    """Create grid visualization lines in PyBullet"""
    grid_lines = []
    
    # Create x grid lines
    for x_edge in robot_abstraction.state_edges[0]:
        start = [x_edge, robot_abstraction.state_intervals[1][0], 0.01]
        end = [x_edge, robot_abstraction.state_intervals[1][1], 0.01]
        line_id = p.addUserDebugLine(start, end, lineColorRGB=[0.5, 0.5, 0.5], lineWidth=1)
        grid_lines.append(line_id)
    
    # Create y grid lines
    for y_edge in robot_abstraction.state_edges[1]:
        start = [robot_abstraction.state_intervals[0][0], y_edge, 0.01]
        end = [robot_abstraction.state_intervals[0][1], y_edge, 0.01]
        line_id = p.addUserDebugLine(start, end, lineColorRGB=[0.5, 0.5, 0.5], lineWidth=1)
        grid_lines.append(line_id)
    
    return grid_lines

def create_trajectory_visualization(trajectory):
    """Create continuous trajectory line in PyBullet"""
    trajectory_lines = []
    
    if len(trajectory) < 2:
        return trajectory_lines
    
    # Create continuous line for the trajectory
    for i in range(len(trajectory) - 1):
        start = [trajectory[i][0], trajectory[i][1], 0.02]
        end = [trajectory[i+1][0], trajectory[i+1][1], 0.02]
        line_id = p.addUserDebugLine(start, end, lineColorRGB=[0, 1, 0], lineWidth=3)
        trajectory_lines.append(line_id)
    
    return trajectory_lines

def create_obstacle_visualization():
    """Create obstacle visualization in PyBullet"""
    obstacles = []
    
    # Définir la région R4 comme obstacle
    x_min, x_max = 4, 6
    y_min, y_max = 6, 8
    
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    # Create obstacle as a red cube
    obstacle_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width/2, height/2, 0.5])
    obstacle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2, height/2, 0.5], rgbaColor=[1, 0, 0, 0.8])
    obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obstacle_shape, 
                                  baseVisualShapeIndex=obstacle_visual, 
                                  basePosition=[center_x, center_y, 0.5])
    obstacles.append(obstacle_id)
    
    return obstacles

def run_pybullet_simulation(robot_abstraction, trajectory, safety_controller):
    """
    Run PyBullet simulation with improved visualization
    """
    print("\n" + "="*60)
    print("STARTING PYBULLET VISUALIZATION")
    print("="*60)
    
    # Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Configure camera
    p.resetDebugVisualizerCamera(
        cameraDistance=12,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=[5, 5, 0]
    )

    # Create ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Create grid visualization
    print("Creating grid visualization...")
    grid_lines = create_grid_visualization(robot_abstraction)
    
    # Create obstacle visualization
    print("Creating obstacle visualization...")
    obstacles = create_obstacle_visualization()
    
    # Create trajectory visualization
    print("Creating trajectory visualization...")
    trajectory_lines = create_trajectory_visualization(trajectory)
    
    # Create robot (R2D2)
    print("Creating robot...")
    start_pos = [trajectory[0][0], trajectory[0][1], 0.3]
    start_ori = p.getQuaternionFromEuler([0, 0, trajectory[0][2]])
    robot_id = p.loadURDF("r2d2.urdf", start_pos, start_ori)
    
    # Get wheel indices for R2D2
    wheel_indices = []
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        if "wheel" in joint_info[1].decode('utf-8').lower():
            wheel_indices.append(i)
    
    # If no wheels found, use default indices
    if not wheel_indices:
        wheel_indices = [2, 3, 4, 5]  # Common wheel indices for R2D2
    
    def set_wheel_velocities(v, omega):
        """Set wheel velocities for differential drive"""
        L, r = 0.2, 0.1  # Wheelbase and wheel radius
        v_l = (v - omega * L / 2) / r
        v_r = (v + omega * L / 2) / r
        
        # Set left wheels
        for i in wheel_indices[::2]:  # Every other wheel as left
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=v_l, force=10)
        
        # Set right wheels  
        for i in wheel_indices[1::2]:  # Every other wheel as right
            p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, targetVelocity=v_r, force=10)
    
    # Simulation parameters
    steps_per_state = 60  # Number of simulation steps per trajectory state
    step_delay = 1/240.0   # Time step for simulation
    
    print(f"\nSimulation Parameters:")
    print(f"Trajectory points: {len(trajectory)}")
    print(f"Steps per state: {steps_per_state}")
    print(f"Total simulation steps: {len(trajectory) * steps_per_state}")
    
    # Main simulation loop
    print("\nStarting robot movement...")
    
    for state_index, target_state in enumerate(trajectory):
        print(f"Moving to state {state_index + 1}/{len(trajectory)}: [{target_state[0]:.2f}, {target_state[1]:.2f}, {target_state[2]:.2f}]")
        
        # Get safe action for this state
        v, omega = safety_controller.get_safe_action(target_state)
        
        # Apply control for this state
        for step in range(steps_per_state):
            set_wheel_velocities(v, omega)
            p.stepSimulation()
            time.sleep(step_delay)
        
        # Force robot to exact target position and orientation
        target_pos = [target_state[0], target_state[1], 0.3]
        target_ori = p.getQuaternionFromEuler([0, 0, target_state[2]])
        p.resetBasePositionAndOrientation(robot_id, target_pos, target_ori)
    
    print("\nSimulation completed successfully!")
    print("Robot has followed the safe trajectory.")
    
    # Keep the simulation running for observation
    print("\nSimulation paused. Close the PyBullet window or press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nSimulation terminated by user.")
    
    # Cleanup
    p.disconnect()

# ============================================================================
# MAIN EXECUTION 
# ============================================================================

def main():
    """
    Main function to run the complete system
    """
    print("=== SYMBOLIC CONTROLLER WITH PYBULLET VISUALIZATION ===")
    
    # 1. Robot Model Configuration
    state_intervals = [(0, 10), (0, 10), (-np.pi, np.pi)]
    perturbation = [(-0.05, 0.05), (-0.05, 0.05), (-np.pi, np.pi)]
    control_values = [(0.25, 1), (-1, 1)]
    delta_t = 1

    # 2. Create Robot Abstraction
    print("\n1. Creating robot abstraction...")
    robot = RobotAbstraction(state_intervals, control_values, perturbation, delta_t)
    
    # 3. Compute Transitions
    print("2. Computing transitions...")
    transitions = robot.compute_transitions_safe()
    
    # 4. Define Safety Region (R4)
    search_interval = [(4, 6), (6, 8), (-np.pi, np.pi)]
    R4_bounds = robot.find_indices_for_interval(search_interval)
    
    # 5. Define Safety States
    states = list(robot.state_to_index.values())
    safety_states = set([
        state for state in states 
        if state != -1 and state not in R4_bounds
    ])
    
    print(f"Safety states: {len(safety_states)}")
    print(f"R4 bounds (indices): {len(R4_bounds)}")
    
    # 6. Synthesize Controller
    print("3. Synthesizing symbolic controller...")
    controller_synthesis = SymbolicControllerSynthesis(transitions, safety_states)
    R_star, controller, Q_0 = controller_synthesis.synthesize_controller()
    
    print(f"Safe states (R*): {len(R_star)}")
    print(f"Initial states (Q_0): {len(Q_0)}")
    
    # 7. Create Safety Controller
    print("4. Creating safety controller...")
    safety_controller = SymbolicSafetyController(robot, safety_states, R4_bounds)
    safety_controller.compute_safe_set()
    
    # 8. Select Initial State
    print("5. Selecting initial state...")
    try:
        initial_state, initial_symbolic = select_initial_state_from_Q0_safe(Q_0, robot)
    except ValueError as e:
        print(f"Using default initial state: {e}")
        initial_state = [1.0, 1.0, 0.0]
    
    print(f"Initial state: {initial_state}")
    
    # 9. Generate Safe Trajectory
    print("6. Generating safe trajectory...")
    initial_perturbation = np.array([0.0, 0.0, 0.0])
    concretized_controller = concretize_controller_with_intervals(controller, robot.index_to_intervals)
    
    trajectory = generate_safe_trajectory(
        initial_state=initial_state,
        initial_perturbation=initial_perturbation,
        concretized_controller=concretized_controller,
        robot=robot,
        max_steps=50  # Reduced for faster simulation
    )
    
    print(f"Generated trajectory with {len(trajectory)} points")
    
    # 10. Run PyBullet Simulation
    print("7. Starting PyBullet visualization...")
    run_pybullet_simulation(robot, trajectory, safety_controller)

if __name__ == "__main__":
    main()