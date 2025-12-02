from dataclasses import dataclass
from typing import Sequence, cast
import numpy as np

from dg_commons import DgSampledSequence, PlayerName
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters

from satellite_docking.agent.planner import SatellitePlanner
from satellite_docking.simulation.goal import SpaceshipTarget, DockingTarget
from satellite_docking.simulation.utils_params import PlanetParams, AsteroidParams
from satellite_docking.simulation.utils_plot import plot_traj


class Config:
    PLOT = False
    VERBOSE = False


@dataclass(frozen=True)
class MyAgentParams:
    """
    Agent parameters.
    """

    # thresholds to decide when to call the planner again
    max_pos_err: float = 1.0  # [m]
    max_vel_err: float = 0.5  # [m/s]
    max_yaw_err: float = np.deg2rad(20.0)  # [rad]
    min_remaining_horizon: float = 3.0  # [s] replan when close to horizon end
    max_replan_interval: float = 8.0  # [s] safety periodic replan

    # gains for feedback around the feedforward command
    kp_pos: float = 1.0  # forward position error -> thrust sum
    kp_vel: float = 0.5  # forward velocity error -> thrust sum
    kp_yaw: float = 2.0  # heading error -> thrust difference
    kd_yaw: float = 0.5  # yaw rate error -> thrust difference


class SatelliteAgent(Agent):
    """
    This is the PDM4AR agent.
    """

    init_state: SatelliteState
    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    goal_state: DynObstacleState

    cmds_plan: DgSampledSequence[SatelliteCommands]
    state_traj: DgSampledSequence[SatelliteState]
    myname: PlayerName
    planner: SatellitePlanner
    goal: PlanningGoal
    static_obstacles: Sequence[StaticObstacle]
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: MyAgentParams

    def __init__(
        self,
        init_state: SatelliteState,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
    ):
        """
        Initializes the agent.
        This method is called by the simulator only before the beginning of each simulation.
        Provides the SatelliteAgent with information about its environment, i.e. planet and satellite parameters and its initial position.
        """
        self.actual_trajectory = []
        self.init_state = init_state
        self.planets = planets
        self.asteroids = asteroids
        self.params = MyAgentParams()
        self.last_plan_time = 0.0

    def on_episode_init(self, init_sim_obs: InitSimObservations):
        """
        This method is called by the simulator only at the beginning of each simulation.
        We suggest to compute here an initial trajectory/node graph/path, used by your planner to navigate the environment.

        Do **not** modify the signature of this method.

        the time spent in this method is **not** considered in the score.
        """
        self.myname = init_sim_obs.my_name
        assert isinstance(init_sim_obs.model_geometry, SatelliteGeometry)
        self.sg = init_sim_obs.model_geometry
        assert isinstance(init_sim_obs.model_params, SatelliteParameters)
        self.sp = init_sim_obs.model_params
        assert isinstance(init_sim_obs.goal, SpaceshipTarget | DockingTarget)
        # make sure you consider both types of goals accordingly
        # (Docking is a subclass of SpaceshipTarget and may require special handling
        # to take into account the docking structure)
        self.goal_state = init_sim_obs.goal.target
        self.goal = init_sim_obs.goal

        # Extract static obstacles for boundary extraction
        if init_sim_obs.dg_scenario is not None:
            self.static_obstacles = list(init_sim_obs.dg_scenario.static_obstacles)
        else:
            self.static_obstacles = []

        # Initialize planner with init and goal states
        # compute initial trajectory guess here as the time is not yet counted
        self.planner = SatellitePlanner(
            planets=self.planets,
            asteroids=self.asteroids,
            sg=self.sg,
            sp=self.sp,
            init_state=self.init_state,
            goal_state=self.goal_state,
            goal=self.goal,
            static_obstacles=self.static_obstacles,
        )

        # Plot docking station (this is optional, for better visualization)
        if Config.PLOT and isinstance(init_sim_obs.goal, DockingTarget):
            A, B, C, A1, A2, _ = init_sim_obs.goal.get_landing_constraint_points()
            init_sim_obs.goal.plot_landing_points(A, B, C, A1, A2)

        # Compute initial trajectory
        try:
            self.cmds_plan, self.state_traj = self.planner.compute_trajectory(self.init_state, self.goal_state)
            self.last_plan_time = 0.0  # Trajectory starts at t=0
        except Exception as e:
            print(f"Initial planning failed: {e}")

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_commands(self, sim_obs: SimObservations) -> SatelliteCommands:
        """
        This method is called by the simulator at every simulation time step. (0.1 sec)
        We suggest to perform two tasks here:
         - Track the computed trajectory (open or closed loop)
         - Plan a new trajectory if necessary
         (e.g., our tracking is deviating from the desired trajectory, the obstacles are moving, etc.)

        NOTE: this function is not run in real time meaning that the simulation is stopped when the function is called.
        Thus the time efficiency of the replanning is not critical for the simulation.
        However the average time spent in get_commands is still considered in the score.

        Do **not** modify the signature of this method.
        """

        # Current and expected states
        current_state = cast(SatelliteState, sim_obs.players[self.myname].state)
        self.actual_trajectory.append(current_state)

        # Use trajectory-relative time (time since last plan)
        traj_time = float(sim_obs.time) - self.last_plan_time
        expected_state = self.state_traj.at_interp(traj_time)

        # ZeroOrderHold
        # cmds = self.cmds_plan.at_or_previous(traj_time)
        # FirstOrderHold
        cmds = self.cmds_plan.at_interp(traj_time)

        # Plotting the trajectory
        if Config.PLOT:
            # Save history every 5.0s (sim_time * 10 % 50 == 0)
            save_history = int(10 * sim_obs.time) % 50 == 0

            if Config.VERBOSE and save_history:
                # start_x = self.state_traj._values[0].x
                # print(f"[DEBUG Plot] Plotting traj starting at x={start_x:.2f}")
                pass

            plot_traj(
                self.state_traj,
                self.actual_trajectory,
                planets=self.planets,
                asteroids=self.asteroids,
                start=self.init_state,
                goal=self.goal_state,
                sim_time=float(sim_obs.time),
                save_history=save_history,
                u=(cmds.F_left, cmds.F_right),
                sg=self.sg,
            )

        ## Adaptive replanning scheme

        # Adaptive tolerance based on distance to both start and goal
        # dist_from_start = np.linalg.norm(current_pos - init_pos)
        # dist_to_goal = np.linalg.norm(current_pos - goal_pos)

        # Replanning heuristic
        if self._should_replan(sim_obs, current_state, expected_state):
            if Config.VERBOSE:
                print(f"[Agent] Recalling planner... (last plan was {traj_time:.2f}s ago)")
            self.cmds_plan, self.state_traj = self.planner.compute_trajectory(current_state, self.goal_state)
            self.last_plan_time = float(sim_obs.time)

            # After replanning, reset traj_time and recompute expected state
            traj_time = 0.0
            expected_state = self.state_traj.at_interp(traj_time)

        # Controller
        cmds = self.cmds_plan.at_interp(traj_time)

        # Potential feedback terms
        if Config.VERBOSE:
            print(
                f"[Agent] t={sim_obs.time:5.2f} pos=({current_state.x:7.2f}, {current_state.y:7.2f}) "
                f"expected=({expected_state.x:7.2f}, {expected_state.y:7.2f}) "
                f"u=(L:{cmds.F_left:6.2f}, R:{cmds.F_right:6.2f}) "
            )

        return cmds  # SatelliteCommands(F_left=1, F_right=1)  # Constant commands

    def _should_replan(
        self,
        sim_obs: SimObservations,
        current_state: SatelliteState,
        expected_state: SatelliteState,
    ) -> bool:
        cfg = self.params

        # 1) tracking error (tighten thresholds when close to goal)
        goal_pos = np.array([self.goal_state.x, self.goal_state.y])
        current_pos = np.array([current_state.x, current_state.y])
        dist_to_goal = np.linalg.norm(current_pos - goal_pos)

        # Tighten tolerances when approaching goal
        if dist_to_goal < 5.0:
            if Config.VERBOSE:
                print(f"[Agent] Tightening replanning thresholds (dist_to_goal={dist_to_goal:.1f}m)")
            scale = max(0.2, dist_to_goal / 5.0)  # Scale down to 20% of original thresholds
            pos_threshold = cfg.max_pos_err * scale
            vel_threshold = cfg.max_vel_err * scale
            yaw_threshold = cfg.max_yaw_err * scale
        else:
            pos_threshold = cfg.max_pos_err
            vel_threshold = cfg.max_vel_err
            yaw_threshold = cfg.max_yaw_err

        pos_err = np.linalg.norm(
            [
                expected_state.x - current_state.x,
                expected_state.y - current_state.y,
            ]
        )
        vel_err = np.linalg.norm(
            [
                expected_state.vx - current_state.vx,
                expected_state.vy - current_state.vy,
            ]
        )
        yaw_err = self._wrap_angle(expected_state.psi - current_state.psi)

        if pos_err > pos_threshold:
            if Config.VERBOSE:
                print(
                    f"[Agent] Replan: position error {pos_err:.2f} > {pos_threshold:.2f} (dist_to_goal={dist_to_goal:.1f}m)"
                )
            return True

        if vel_err > vel_threshold:
            if Config.VERBOSE:
                print(
                    f"[Agent] Replan: velocity error {vel_err:.2f} > {vel_threshold:.2f} (dist_to_goal={dist_to_goal:.1f}m)"
                )
            return True

        if abs(yaw_err) > yaw_threshold:
            if Config.VERBOSE:
                print(
                    f"[Agent] Replan: yaw error {yaw_err:.2f} > {yaw_threshold:.2f} (dist_to_goal={dist_to_goal:.1f}m)"
                )
            return True

        return False
