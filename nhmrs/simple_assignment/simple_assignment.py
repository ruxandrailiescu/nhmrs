"""Simple Assignment v0 - NHMRS Task Allocation Environment.

This module implements a PettingZoo ParallelEnv for multi-agent task allocation
where N non-holonomic mobile robots must visit M landmarks (tasks). The environment
supports arbitrary kinematic models (unicycle, differential drive, Ackermann) and
provides a modular reward function that adapts to different N/M ratios.

Key Features:
    - Kinematic models: Pluggable dynamics (unicycle, diff-drive, ackermann)
    - Reward modes: 'simple' (fast), 'balanced' (default), 'patrol' (N < M)
    - Auto-zoom rendering: Camera automatically fits all entities
    - PettingZoo ParallelEnv API: Compatible with standard MARL libraries

Environment Structure:
    - Agents: Non-holonomic robots with [x, y, θ] state
    - Landmarks: Static task locations [x, y]
    - Observations: own_state + all_landmark_positions + other_agent_states
    - Actions: Kinematic controls (e.g., [v, ω] for unicycle)
    - Rewards: Modular multi-component (see SimpleAssignmentReward)

Example:
    >>> from nhmrs import simple_assignment_v0
    >>> env = simple_assignment_v0.env(render_mode="human")
    >>> obs, info = env.reset(seed=42)
    >>> actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    >>> obs, rewards, terminated, truncated, info = env.step(actions)
"""

import numpy as np
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo import AECEnv
from gymnasium import spaces
from gymnasium.utils import EzPickle
from scipy.optimize import linear_sum_assignment

# Core NHMRS imports
from nhmrs._nhmrs_utils.core import World, Agent, Landmark
from nhmrs._nhmrs_utils.scenario import BaseScenario
from nhmrs._nhmrs_utils.kinematics import UnicycleKinematics

# Simple assignment specific imports
from .simple_assignment_reward import SimpleAssignmentReward, SimpleSpreadReward

# Rendering imports (optional)
try:
    from nhmrs._nhmrs_utils.rendering import (
        Viewer, Transform, make_circle, make_polygon, make_agent_geom
    )
    RENDERING_AVAILABLE = True
except ImportError:
    RENDERING_AVAILABLE = False


def make_env(raw_env):
    """Wrap raw environment with standard PettingZoo wrappers.
    
    Creates a factory function that instantiates the environment with given kwargs.
    This is the standard PettingZoo pattern for environment registration.
    
    Args:
        raw_env: The raw ParallelEnv class to wrap
        
    Returns:
        Callable: Factory function that creates environment instances
    """
    def env_fn(**kwargs):
        return raw_env(**kwargs)
    return env_fn


class raw_env(ParallelEnv, EzPickle):
    """Raw PettingZoo ParallelEnv for task allocation with non-holonomic robots.
    
    This is the core environment implementing the PettingZoo parallel API.
    N agents (default 3) navigate to M landmarks (default 4) arranged in a circle.
    Each agent receives its own state, all landmark positions, and other agents' states.
    
    The environment uses a World container (from core.py) to manage agent states
    and apply kinematic updates each step. Rewards are computed by a Scenario object
    that delegates to a SimpleAssignmentReward instance.
    
    Observation Space (per agent):
        Box with shape (state_dim + 2*M + state_dim*(N-1),)
        - Own state: [x, y, θ] (or [x, y] for holonomic)
        - Landmark positions: [x1, y1, x2, y2, ..., xM, yM]
        - Other agents: [x_j, y_j, θ_j for j != i]
    
    Action Space (per agent):
        Box defined by kinematics.get_action_bounds()
        - Unicycle: [v, ω] linear and angular velocity
        - Differential drive: [v_left, v_right] wheel velocities
        - Ackermann: [v, δ] speed and steering angle
    
    Attributes:
        scenario (Scenario): Manages world creation, reset, reward, and observations
        world (World): Container for agents, landmarks, and step logic
        max_steps (int): Episode truncation limit
        render_mode (str | None): "human", "rgb_array", or None
        viewer (Viewer | None): Pygame rendering window (created on first render)
    
    Args:
        scenario: Scenario object for world configuration. If None, uses default
            Scenario() with 3 agents and 4 landmarks.
        render_mode: Rendering mode - "human" for window, "rgb_array" for numpy
        max_steps: Maximum steps before episode truncation (default 500)
        kinematics: KinematicsModel instance shared by all agents (default UnicycleKinematics)
        agent_size: Visual radius for rendering agents (default 0.075)
    """
    
    metadata = {
        "name": "simple_assignment_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(self, scenario=None, render_mode=None, max_steps=500,
                 kinematics=None, agent_size=0.075, n_agents=3, n_landmarks=4,
                 observe_relative=True):
        EzPickle.__init__(
            self,
            scenario=scenario,
            render_mode=render_mode,
            max_steps=max_steps,
            kinematics=kinematics,
            agent_size=agent_size,
            n_agents=n_agents,
            n_landmarks=n_landmarks,
            observe_relative=observe_relative,
        )
        super().__init__()
        
        # Create scenario and world using core abstractions
        self.scenario = scenario if scenario is not None else Scenario(observe_relative=observe_relative)
        self.world = self.scenario.make_world(n_agents=n_agents, n_landmarks=n_landmarks)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.agent_size = agent_size
        
        # Set default kinematics if not provided
        default_kinematics = kinematics if kinematics is not None else UnicycleKinematics()
        
        # Ensure all agents have kinematics
        for agent in self.world.agents:
            if agent.kinematics is None:
                agent.kinematics = default_kinematics
        
        # PettingZoo attributes
        self.possible_agents = [agent.name for agent in self.world.agents]
        self.agents = []
        
        # Spaces (based on first agent's kinematics model)
        first_agent = self.world.agents[0]
        action_low, action_high = first_agent.kinematics.get_action_bounds()
        self._action_space_template = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # Observation: own state + landmarks + other agents
        state_dim = first_agent.kinematics.state_dim
        obs_dim = state_dim + 2 * len(self.world.landmarks) + state_dim * (len(self.world.agents) - 1)
        self._observation_space_template = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_spaces = {}
        self.observation_spaces = {}
        
        # State
        self._rng = np.random.RandomState()
        self.t = 0
        
        # Rendering
        self.viewer = None
        self._agent_geoms = []
        self._agent_transforms = []
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.
        
        Steps:
            1) Seed the RNG if provided
            2) Reset agent list and spaces
            3) Call scenario.reset_world() to place agents/landmarks
            4) Reset timestep counter
            5) Compute initial observations
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused, for PettingZoo compatibility)
            
        Returns:
            tuple: (observations, infos)
                - observations: Dict mapping agent_name -> observation array
                - infos: Dict mapping agent_name -> info dict (empty)
        """
        if seed is not None:
            self._rng.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.action_spaces = {a: self._action_space_template for a in self.agents}
        self.observation_spaces = {a: self._observation_space_template for a in self.agents}
        
        # Reset world using core abstractions
        self.scenario.reset_world(self.world, self._rng)
        self.t = 0
        
        obs = self._get_obs()
        info = {a: {} for a in self.agents}
        return obs, info
    
    def step(self, actions):
        """Advance the environment by one timestep.
        
        Steps:
            1) Set action.u for each agent from the actions dict
            2) Call world.step() to apply kinematics updates
            3) Increment timestep counters (env and world)
            4) Compute rewards via scenario.reward()
            5) Check termination (never terminates early, only truncates at max_steps)
            6) Collect observations
        
        Args:
            actions: Dict mapping agent_name -> action array (kinematic controls)
        
        Returns:
            tuple: (observations, rewards, terminated, truncated, infos)
                - observations: Dict[agent_name, obs_array]
                - rewards: Dict[agent_name, float]
                - terminated: Dict[agent_name, bool] (always False, no early termination)
                - truncated: Dict[agent_name, bool] (True when t >= max_steps)
                - infos: Dict[agent_name, dict] (empty)
        """
        # Set actions for each agent
        for agent_obj in self.world.agents:
            if agent_obj.name in actions:
                agent_obj.action.u = actions[agent_obj.name]
        
        # Step the world (applies kinematics for all agents)
        self.world.step()
        
        # Update timesteps
        self.t += 1
        self.world.t = self.t  # Sync world timestep for reward computation
        
        # Compute rewards using scenario
        rewards = {}
        for agent_obj in self.world.agents:
            rewards[agent_obj.name] = self.scenario.reward(agent_obj, self.world)
        
        # Check termination
        terminated = {a: False for a in self.agents}
        truncated = {a: self.t >= self.max_steps for a in self.agents}
        
        # Clear agents if episode done
        if all(truncated.values()):
            self.agents = []
        
        obs = self._get_obs()
        info = {a: {} for a in self.agents if a in self.agents}
        
        return obs, rewards, terminated, truncated, info
    
    def _get_obs(self):
        """Collect observations for all active agents.
        
        Delegates to scenario.observation() which constructs:
            [own_x, own_y, own_θ, lm1_x, lm1_y, ..., lmM_x, lmM_y,
             other1_x, other1_y, other1_θ, ..., otherN_x, otherN_y, otherN_θ]
        
        Returns:
            Dict[str, np.ndarray]: Observations keyed by agent name
        """
        obs = {}
        for agent_obj in self.world.agents:
            if agent_obj.name in self.agents:
                obs[agent_obj.name] = self.scenario.observation(agent_obj, self.world)
        return obs
    
    def render(self):
        """Render the current state to screen or RGB array.
        
        Steps:
            1) Create viewer if first call (lazy initialization)
            2) Compute auto-zoom bounds to fit all agents and landmarks
            3) Update agent transforms (position and orientation)
            4) Draw landmarks as one-time geometries
            5) Render and return (None for human mode, array for rgb_array mode)
        
        The camera automatically zooms to fit all entities with 1.0 world unit margin.
        Agents are drawn as colored arrowheads with gray collision circles.
        Landmarks are drawn as red circles.
        
        Returns:
            np.ndarray | None: RGB array (H, W, 3) if render_mode="rgb_array", else None
        """
        if self.render_mode is None:
            return None
        
        if self.viewer is None:
            self.viewer = Viewer(700, 700)
            self._setup_rendering()
        
        # Auto-zoom to fit all entities
        all_pos = []
        for agent in self.world.agents:
            all_pos.append(agent.state.p_pos)
        for landmark in self.world.landmarks:
            all_pos.append(landmark.state.p_pos)
        
        if all_pos:
            all_pos = np.array(all_pos)
            min_x, min_y = all_pos.min(axis=0)
            max_x, max_y = all_pos.max(axis=0)
            
            # Add padding
            margin = 1.0
            min_x -= margin
            max_x += margin
            min_y -= margin
            max_y += margin
            
            # Ensure aspect ratio (square viewport)
            cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
            w, h = max_x - min_x, max_y - min_y
            size = max(w, h) / 2
            
            self.viewer.set_bounds(cx - size, cx + size, cy - size, cy + size)
        
        # Update agent transforms
        for i, agent in enumerate(self.world.agents):
            pos = agent.state.p_pos
            theta = agent.state.p_orient
            self._agent_transforms[i].set_translation(pos[0], pos[1])
            self._agent_transforms[i].set_rotation(theta)
        
        # Draw landmarks
        for landmark in self.world.landmarks:
            circ = make_circle(landmark.size)
            circ.set_color(*landmark.color)
            transform = Transform()
            transform.set_translation(landmark.state.p_pos[0], landmark.state.p_pos[1])
            circ.add_attr(transform)
            self.viewer.add_onetime(circ)
        
        return self.viewer.render(return_rgb_array=(self.render_mode == "rgb_array"))
    
    def _setup_rendering(self):
        """Create persistent agent geometries (called once on first render).
        
        For each agent:
            1) Create collision circle (gray outline) and arrowhead (agent color)
            2) Both share a Transform that will be updated each frame
            3) Add to viewer (circle first so arrow draws on top)
            4) Store transform reference for position/rotation updates
        
        Agents are rendered as colored arrowheads inscribed in gray circles.
        The arrowhead shows orientation, the circle shows collision radius.
        """
        self._agent_geoms = []
        self._agent_transforms = []
        self._agent_circles = []
        
        for agent in self.world.agents:
            # Use shared agent geometry creator from rendering utilities
            collision_circle, arrow, transform = make_agent_geom(
                agent.size, agent.color
            )
            
            # Add geometries to viewer (circle first, then arrow - draw order matters)
            self.viewer.add_geom(collision_circle)
            self.viewer.add_geom(arrow)
            
            # Store references for updates
            self._agent_circles.append(collision_circle)
            self._agent_geoms.append(arrow)
            self._agent_transforms.append(transform)
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]


env = make_env(raw_env)
parallel_env = raw_env


class Scenario(BaseScenario):
    """Scenario for task allocation with configurable reward modes.
    
    Manages world creation, reset, observation, and reward computation.
    The scenario can use either SimpleAssignmentReward (explicit assignment
    via Hungarian algorithm) or SimpleSpreadReward (MPE2-style cooperative
    spreading with occupancy + collision).
    
    Available reward modes:
        - 'spread': MPE2-style occupancy + collision (default)
        - 'simple': Assignment + collision only (fast learning)
        - 'balanced': Full multi-component assignment reward
        - 'patrol': Emphasizes coverage and idleness for N < M scenarios
    
    Attributes:
        reward_mode (str): One of 'spread', 'simple', 'balanced', 'patrol'
        reward_computer (SimpleAssignmentReward | SimpleSpreadReward): Computes per-agent rewards
    """
    
    def __init__(self, reward_mode='spread', observe_relative=True):
        """Initialize scenario with reward configuration.
        
        Args:
            reward_mode: Reward weighting preset:
                - 'spread': MPE2-style occupancy + collision (default)
                - 'simple': Assignment + collision only (fast learning)
                - 'balanced': Full reward with all components
                - 'patrol': Optimized for N < M persistent coverage
            observe_relative: If True, landmark and other-agent positions are
                expressed as offsets from the observing agent (translation-invariant).
                If False, raw world-frame coordinates are used. Default True.
        """
        self.reward_mode = reward_mode
        self.observe_relative = observe_relative
        self.reward_computer = None
    
    def make_world(self, n_agents=3, n_landmarks=4):
        """Create world with agents and landmarks.
        
        Constructs a World object with N agents and M landmarks. Agents are
        assigned sequential names (agent_0, agent_1, ...) and distinct colors.
        Landmarks are given sequential names (landmark_0, landmark_1, ...).
        
        Kinematics are not set here - the environment will assign them during init.
        
        Args:
            n_agents: Number of controllable agents (default 3)
            n_landmarks: Number of task locations (default 4)
            
        Returns:
            World: Populated world ready for reset
        """
        world = World()
        
        # Create agents
        agent_colors = [
            (0.25, 0.50, 0.85),  # blue
            (0.85, 0.85, 0.25),  # yellow
            (0.85, 0.25, 0.25),  # red
            (0.25, 0.85, 0.25),  # green
            (0.85, 0.25, 0.85),  # magenta
            (0.25, 0.85, 0.85),  # cyan
        ]
        
        for i in range(n_agents):
            agent = Agent()
            agent.name = f'agent_{i}'
            agent.size = 0.075
            agent.color = agent_colors[i % len(agent_colors)]
            agent.kinematics = None  # Will be set in environment init
            world.agents.append(agent)
        
        # Create landmarks (tasks)
        for i in range(n_landmarks):
            landmark = Landmark()
            landmark.name = f'landmark_{i}'
            landmark.size = 0.1
            landmark.color = (0.25, 0.85, 0.85)  # cyan
            world.landmarks.append(landmark)
        
        # Initialize reward computer based on mode
        if self.reward_mode == 'simple':
            self.reward_computer = SimpleAssignmentReward(
                weights={
                    'assignment': 1.0,
                    'coverage': 0.0,
                    'collision': 5.0,
                    'idleness': 0.0,
                    'efficiency': 0.0,
                },
                safety_radius=0.3
            )
        elif self.reward_mode == 'patrol':
            self.reward_computer = SimpleAssignmentReward(
                weights={
                    'assignment': 0.5,
                    'coverage': 1.0,
                    'collision': 5.0,
                    'idleness': 1.0,
                    'efficiency': 0.1,
                },
                safety_radius=0.3,
                visit_threshold=0.5
            )
        elif self.reward_mode == 'spread':
            self.reward_computer = SimpleSpreadReward(
                collision_weight=1.0,
                agent_radius=0.075  # Match agent.size from make_world()
            )
        else:  # 'balanced'
            self.reward_computer = SimpleAssignmentReward(
                weights={
                    'assignment': 1.0,
                    'coverage': 0.5,
                    'collision': 5.0,
                    'idleness': 0.3,
                    'efficiency': 0.1,
                },
                safety_radius=0.3,
                visit_threshold=0.5
            )
        
        return world
    
    def reset_world(self, world, np_random):
        """Initialize agent and landmark positions.
        
        Agents are placed randomly in [-1, 1]² with random orientations.
        Landmarks are placed evenly around a circle of radius 3.0.
        
        Also resets the reward computer's internal state (visit tracking, timestep).
        
        Args:
            world: World object to initialize
            np_random: Random number generator (from env._rng)
        """
        # Reset reward computer state
        if self.reward_computer is not None:
            self.reward_computer.reset()
        
        # Initialize agent positions randomly
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, 1, 2).astype(np.float32)
            agent.state.p_orient = np_random.uniform(-np.pi, np.pi)
        
        # Place landmarks in a circle
        for i, landmark in enumerate(world.landmarks):
            angle = 2 * np.pi * i / len(world.landmarks)
            radius = 3.0
            landmark.state.p_pos = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ], dtype=np.float32)
    
    def reward(self, agent, world):
        """Compute reward for a single agent.
        
        Delegates to SimpleAssignmentReward.compute_reward() if available,
        otherwise falls back to simple negative distance to nearest landmark.
        
        Args:
            agent: Agent object with state.p_pos
            world: World with agents, landmarks, and timestep
            
        Returns:
            float: Scalar reward (typically negative, representing cost)
        """
        if self.reward_computer is None:
            # Fallback to simple distance-based reward
            dists = [np.linalg.norm(agent.state.p_pos - lm.state.p_pos) 
                     for lm in world.landmarks]
            return -min(dists)
        
        # Update timestep in reward computer
        if hasattr(world, 't'):
            self.reward_computer.update_timestep(world.t)
        
        # Use modular reward computation
        return self.reward_computer.compute_reward(agent, world)
    
    def observation(self, agent, world):
        """Construct observation vector for one agent.

        Observation structure (observe_relative=True):
            [own_x, own_y, own_θ,                              # 3 values (world frame)
             lm1_Δx, lm1_Δy, ..., lmM_Δx, lmM_Δy,            # 2*M values (relative)
             other1_Δx, other1_Δy, other1_Δθ, ...]            # 3*(N-1) values (relative)

        Observation structure (observe_relative=False):
            [own_x, own_y, own_θ,                              # 3 values (world frame)
             lm1_x, lm1_y, ..., lmM_x, lmM_y,                 # 2*M values (world frame)
             other1_x, other1_y, other1_θ, ...]                # 3*(N-1) values (world frame)

        Total size: 3 + 2*M + 3*(N-1) in both cases.

        Args:
            agent: Agent to observe for
            world: World with agents and landmarks

        Returns:
            np.ndarray: Observation vector, shape (obs_dim,)
        """
        own_state = np.array([
            agent.state.p_pos[0],
            agent.state.p_pos[1],
            agent.state.p_orient
        ], dtype=np.float32)

        if self.observe_relative:
            landmark_pos = np.concatenate([
                lm.state.p_pos - agent.state.p_pos for lm in world.landmarks
            ])
            other_states = []
            for other in world.agents:
                if other is not agent:
                    rel_pos = other.state.p_pos - agent.state.p_pos
                    rel_orient = other.state.p_orient - agent.state.p_orient
                    # Normalize relative orientation to [-π, π]
                    rel_orient = np.arctan2(np.sin(rel_orient), np.cos(rel_orient))
                    other_states.extend([rel_pos[0], rel_pos[1], rel_orient])
        else:
            landmark_pos = np.concatenate([lm.state.p_pos for lm in world.landmarks])
            other_states = []
            for other in world.agents:
                if other is not agent:
                    other_states.extend([
                        other.state.p_pos[0],
                        other.state.p_pos[1],
                        other.state.p_orient
                    ])

        return np.concatenate([own_state, landmark_pos,
                                np.array(other_states, dtype=np.float32)])
