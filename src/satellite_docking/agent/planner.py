import ast
from dataclasses import dataclass, field, replace
import math
from typing import Sequence, Union
from enum import Enum

import cvxpy as cvx
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.seq import DgSampledSequence
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.obstacles_dyn import DynObstacleState
from dg_commons.sim.models.satellite import SatelliteCommands, SatelliteState
from dg_commons.sim.models.satellite_structures import (
    SatelliteGeometry,
    SatelliteParameters,
)
from shapely.geometry import LineString

from satellite_docking.agent.discretization import *
from satellite_docking.simulation.utils_params import PlanetParams, AsteroidParams
import numpy as np
from satellite_docking.simulation.goal import DockingTarget


class PlannerConfig:
    VERBOSE = False
    PLOT = False


class StopCritMode(Enum):
    ABSOLUTE_COST = 1  # $J_{\lambda}(\bar{x}, \bar{u}, \bar{p}) - L_{\lambda}(x^*, u^*, p^*, \hat{\nu}^*) \leq \epsilon$
    STATE_PARAM = 2  # $||p^* - \bar{p}||_q + \max_k ||x_k^* - \bar{x}_k||_q \leq \epsilon$
    RELATIVE_COST = 3  # $J_{\lambda}(\bar{x}, \bar{u}, \bar{p}) - L_{\lambda}(x^*, u^*, p^*, \hat{\nu}^*) \leq \epsilon_r |J_{\lambda}(\bar{x}, \bar{u}, \bar{p})|$


@dataclass(frozen=True)
class SolverParameters:
    """
    Definition space for SCvx parameters in case SCvx algorithm is used.
    Parameters can be fine-tuned by the user.
    """

    # Cvxpy solver parameters
    solver: str = "CLARABEL"  # specify solver to use
    verbose_solver: bool = False  # if True, the optimization steps are shown
    max_iterations: int = 100

    # SCVX parameters (Add paper reference)
    weight_nu_dyn: float = 1e2
    weight_nu_path: float = 1e2
    weight_nu_boundary: float = 1e2
    weight_time: float = 15.0
    weight_distance: float = 3.0 / 5.0
    weight_effort: float = 3.0

    tf_max: float = 60.0

    margin_collision_distance: float = 1.05

    tr_radius: float = 10.0
    min_tr_radius: float = 0.001
    max_tr_radius: float = 50.0
    rho_0: float = 0.0
    rho_1: float = 0.1
    rho_2: float = 0.7
    alpha: float = 0.5
    beta: float = 2.0

    # Discretization constants
    K: int = 50  # number of discretization steps
    N_sub: int = 5  # used inside ode solver inside discretization
    stop_crit: float = 1e-2
    stop_crit_rel: float = 1e-2
    stop_crit_mode: StopCritMode = StopCritMode.RELATIVE_COST  # Default to mode 2
    stop_crit_norm: int = 2  # Norm type q (1, 2) used for mode 2


class SatellitePlanner:
    """
    Satellite Planner.
    """

    planets: dict[PlayerName, PlanetParams]
    asteroids: dict[PlayerName, AsteroidParams]
    satellite: SatelliteDyn
    sg: SatelliteGeometry
    sp: SatelliteParameters
    params: SolverParameters
    init_state: SatelliteState
    goal_state: DynObstacleState
    planning_goal: PlanningGoal

    # Simpy variables
    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    X_bar: NDArray
    U_bar: NDArray
    p_bar: NDArray

    J_old: float
    J_new: float
    L_new: float

    tr_eta: float

    def __init__(
        self,
        planets: dict[PlayerName, PlanetParams],
        asteroids: dict[PlayerName, AsteroidParams],
        sg: SatelliteGeometry,
        sp: SatelliteParameters,
        init_state: SatelliteState,
        goal_state: DynObstacleState,
        goal: PlanningGoal,
        static_obstacles: Sequence[StaticObstacle] | None = None,
    ):
        """
        Pass environment information to the planner.
        """
        self.planets = planets
        self.asteroids = asteroids
        self.sg = sg
        self.sp = sp
        self.init_state = init_state
        self.goal_state = goal_state
        self.planning_goal = goal

        # Extract state bounds from static obstacles (LineString boundary)
        self.state_bounds = self._extract_state_bounds(static_obstacles)

        # Satellite Geometry
        self.sat_radius = np.sqrt((self.sg.w_half + self.sg.w_panel) ** 2 + (self.sg.l_f) ** 2)

        if PlannerConfig.VERBOSE and self.state_bounds is not None:
            print(
                f"[Planner] State bounds extracted: x=[{self.state_bounds['x_min']:.2f}, {self.state_bounds['x_max']:.2f}], "
                f"y=[{self.state_bounds['y_min']:.2f}, {self.state_bounds['y_max']:.2f}]"
            )

        # Solver Parameters
        self.params = SolverParameters()

        # Satellite Dynamics
        self.satellite = SatelliteDyn(self.sg, self.sp)
        self.n_x = self.satellite.n_x
        self.n_u = self.satellite.n_u
        self.n_p = self.satellite.n_p

        # Define dimensions for constraints
        self.n_ic = self.n_x
        self.n_tc = self.n_x
        self.n_s = len(self.planets) + len(self.asteroids)

        # Discretization Method
        # self.integrator = ZeroOrderHold(self.Satellite, self.params.K, self.params.N_sub)
        self.integrator = FirstOrderHold(self.satellite, self.params.K, self.params.N_sub)

        # Check dynamics implementation
        if not self.integrator.check_dynamics():
            raise ValueError("Dynamics check failed.")
        else:
            print("Dynamics check passed.")

        # Variables
        self.variables = self._get_variables()

        # Problem Parameters
        self.problem_parameters = self._get_problem_parameters()

        # Compute initial guess (called on SatellitePlanner init during on_episode_init and thus the time is
        # not counted in the score)
        self.X_bar, self.U_bar, self.p_bar = self.initial_guess()

        # Time scaling factor for effort cost, stored as a scalar to keep the convex problem DCP.
        self.dt_scale = float(self.p_bar[0] / (self.params.K - 1)) if self.n_p > 0 else 1.0

        self.J_old = np.inf
        self.J_new = np.inf
        self.L_new = np.inf

        self.tr_eta = self.params.tr_radius

        # Constraints
        constraints = self._get_constraints()

        # Objective
        objective = self._get_objective()

        # Cvx Optimisation Problem
        self.problem = cvx.Problem(objective, constraints)

        # Check if problem is DDP

    def compute_trajectory(
        self,
        current_state: SatelliteState,
        goal_state: DynObstacleState,
        asteroids: dict[PlayerName, AsteroidParams] | None = None,
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Compute a trajectory from init_state to goal_state.
        Note: init_state and goal_state are already set in __init__.
        This method can be called for replanning with updated states if needed.

        Args:
            init_state (SatelliteState): The initial state of the satellite.
            goal_state (DynObstacleState): The goal state to reach.
            asteroids (dict): Optional updated asteroid parameters (current state).

        Returns:
            tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
                A tuple containing the sequence of commands and the state trajectory.
        """
        # Update states
        self.init_state = current_state
        self.goal_state = goal_state

        # Update asteroids if provided (not in init call)
        if asteroids is not None:
            self.asteroids = asteroids

        if self.asteroids is None:
            self.asteroids = {}

        if PlannerConfig.VERBOSE:
            print(f"[Planner] Recomputing from x={current_state.x:.2f}, y={current_state.y:.2f}")
            print(f"[Planner DEBUG] Asteroids received ({len(self.asteroids)}):")
            for name, params in self.asteroids.items():
                theta = params.orientation
                c, s = np.cos(theta), np.sin(theta)
                vel_world = np.array(
                    [c * params.velocity[0] - s * params.velocity[1], s * params.velocity[0] + c * params.velocity[1]]
                )
                print(
                    f"  {name}: start=({params.start[0]:.2f}, {params.start[1]:.2f}), "
                    f"vel_body=({params.velocity[0]:.2f}, {params.velocity[1]:.2f}), "
                    f"vel_world=({vel_world[0]:.2f}, {vel_world[1]:.2f}), "
                    f"orient={params.orientation:.2f}rad, r={params.radius:.2f}"
                )

        # Reset trust region
        self.tr_eta = self.params.tr_radius

        # 1. Initial Guess
        self.X_bar, self.U_bar, self.p_bar = self.initial_educated_guess()

        # Calculating J_lambda at initial guess
        self.J_old = self._compute_J_lambda(self.X_bar, self.U_bar, self.p_bar)
        final_X, final_U, final_p = self.X_bar, self.U_bar, self.p_bar

        # 2. Iterate
        for i in range(self.params.max_iterations):

            # 3. Convexification
            self._convexification()

            # 6. Solve convex subproblem
            try:
                error = self.problem.solve(
                    verbose=self.params.verbose_solver, solver=self.params.solver, warm_start=True
                )
            except cvx.SolverError:
                print(f"SolverError: {self.params.solver} failed to solve the problem.")
                break

            # Extract new solution
            X_new = self.variables["X"].value  # shape (n_x, K)
            U_new = self.variables["U"].value  # shape (n_u, K)
            p_new = self.variables["p"].value  # shape (n_p,)
            self.J_new = self._compute_J_lambda(X_new, U_new, p_new)

            # Get convexified cost at new solution
            self.L_new = self.problem.value

            final_X, final_U, final_p = X_new, U_new, p_new

            if PlannerConfig.VERBOSE:
                dt_candidate = float(p_new[0] / (self.params.K - 1)) if self.n_p > 0 else 1.0
                print(
                    f"[SCvx] iter {i:02d}: L={self.L_new:.3f} J_old={self.J_old:.3f} J_new={self.J_new:.3f}  "
                    f"dt={dt_candidate:.3f}s eta={self.tr_eta:.3f}"
                )

            # 8. Check convergence
            if self._check_convergence(X_new, U_new, p_new, self.L_new, self.J_new):
                print(f"Converged in {i} iterations.")

                # Save progress for next call
                self.X_bar, self.U_bar, self.p_bar = X_new, U_new, p_new

                break

            # 7. Update trust region
            if self._update_trust_region():
                self.J_old = self.J_new
                self.X_bar, self.U_bar, self.p_bar = X_new, U_new, p_new

        cmds_seq, states_seq = self._extract_seq_from_array(self.X_bar, self.U_bar, self.p_bar)

        return cmds_seq, states_seq

    def initial_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Simple straight-line initial guess for SCvx.
        Linear interpolation between start and goal states.
        SCvx will handle the nonlinear dynamics correction.

        Returns:
            tuple[NDArray, NDArray, NDArray]: Initial guess for states X, controls U, and parameters p.
        """
        K = self.params.K
        X = np.zeros((self.n_x, K))
        U = np.zeros((self.n_u, K))

        # Extract start and goal states
        x0 = self._state_to_vec(self.init_state)
        xf = self._state_to_vec(self.goal_state)

        # Linear interpolation for all states
        for k in range(K):
            alpha = k / (K - 1)
            X[:, k] = (1 - alpha) * x0 + alpha * xf

        # Estimate total time based on distance
        dist = np.linalg.norm(xf[:2] - x0[:2])
        v_max = self.sp.vx_limits[1] if hasattr(self.sp, "vx_limits") else 1.0
        T = max(1.5 * dist / v_max, 5.0) if v_max > 1e-6 else 30.0
        if hasattr(self.params, "tf_max"):
            T = min(T, self.params.tf_max)

        p = np.array([T]) if self.n_p > 0 else np.zeros(self.n_p)

        if PlannerConfig.VERBOSE:
            print(f"[InitGuess] Straight line: dist={dist:.2f}m, T={T:.2f}s")

        return X, U, p

    def initial_educated_guess(self) -> tuple[NDArray, NDArray, NDArray]:
        """
        Warm start using the previous solution (X_bar, U_bar, p_bar).

        Strategy:
        1. Find where we are on the old trajectory (time-shift)
        2. Extract the "tail" from that point and re-interpolate to K knots
        3. Adjust time estimate and enforce boundary conditions

        If we're far off-track, fall back to straight-line initial_guess().
        """
        K = self.params.K

        X_old = self.X_bar.copy()
        U_old = self.U_bar.copy()
        p_old = self.p_bar.copy()

        x_curr = self._state_to_vec(self.init_state)
        x_goal = self._state_to_vec(self.goal_state)

        # --- 1) Find closest knot on old trajectory ---
        dists = np.linalg.norm(X_old[:2, :] - x_curr[:2, np.newaxis], axis=0)
        k_closest = int(np.argmin(dists))
        min_dist = dists[k_closest]

        # If far off-track, the old trajectory isn't useful
        if min_dist > 2.0:
            if PlannerConfig.VERBOSE:
                print(f"[EduGuess] Off-track ({min_dist:.2f}m), using straight-line")
            return self.initial_guess()

        # --- 2) Extract tail and re-interpolate to K knots ---
        k_closest = min(k_closest, K - 2)  # ensure at least 2 points in tail
        K_tail = K - k_closest

        X_tail = X_old[:, k_closest:]
        U_tail = U_old[:, k_closest:]

        t_old = np.linspace(0, 1, K_tail)
        t_new = np.linspace(0, 1, K)

        X = np.zeros((self.n_x, K))
        U = np.zeros((self.n_u, K))
        for i in range(self.n_x):
            X[i, :] = np.interp(t_new, t_old, X_tail[i, :])
        for i in range(self.n_u):
            U[i, :] = np.interp(t_new, t_old, U_tail[i, :])

        # --- 3) Translate and enforce boundary conditions ---
        delta_xy = x_curr[:2] - X[:2, 0]
        X[0, :] += delta_xy[0]
        X[1, :] += delta_xy[1]

        X[:, 0] = x_curr
        X[:, -1] = x_goal
        U[:, 0] = 0.0
        U[:, -1] = 0.0

        # Blend first few knots
        for k in range(1, min(3, K)):
            w = 1.0 - k / 3
            X[2:, k] = w * x_curr[2:] + (1.0 - w) * X[2:, k]

        # --- 4) Adjust time ---
        if self.n_p > 0:
            T_old = p_old[0]
            T_new = np.clip(T_old * K_tail / K, 1.0, self.params.tf_max)
            p = np.array([T_new])
            if PlannerConfig.VERBOSE:
                print(f"[EduGuess] k={k_closest}, T: {T_old:.1f}s -> {T_new:.1f}s")
        else:
            p = np.zeros(self.n_p)
        return X, U, p

    def _set_goal(self):
        """
        Sets goal for SCvx.
        """
        self.goal = cvx.Parameter((6, 1))
        pass

    def _get_variables(self) -> dict:
        """
        Define optimisation variables for SCvx.
        """
        variables = {
            "X": cvx.Variable((self.n_x, self.params.K)),
            "U": cvx.Variable((self.n_u, self.params.K)),
            "p": cvx.Variable(self.n_p, nonneg=True),
            "nu_dyn": cvx.Variable((self.n_x, self.params.K - 1)),
            "nu_s": cvx.Variable((self.n_s, self.params.K), nonneg=True),  # slack for state constraints
            "nu_ic": cvx.Variable(self.n_ic),  # slack for initial condition
            "nu_tc": cvx.Variable(self.n_tc),  # slack for terminal condition
            # "nu_docking": cvx.Variable((2, self.params.K), nonneg=True),  # slack for docking (orientation + centering)
        }

        return variables

    def _get_problem_parameters(self) -> dict:
        """
        Define problem parameters for SCvx.
        """
        n_x, n_u, n_p = self.n_x, self.n_u, self.n_p
        n_s, n_ic, n_tc = self.n_s, self.n_ic, self.n_tc
        K = self.params.K

        problem_parameters = {
            # boundary
            "init_state": cvx.Parameter(n_x),
            "goal_state": cvx.Parameter(n_x),
            # linearized, discretized dynamics (FOH form)
            # A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar as flattened matrices
            "A_bar": cvx.Parameter((n_x * n_x, K - 1)),
            "B_plus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "B_minus_bar": cvx.Parameter((n_x * n_u, K - 1)),
            "F_bar": cvx.Parameter((n_x * n_p, K - 1)),
            "r_bar": cvx.Parameter((n_x, K - 1)),
            # collision constraints
            "C": cvx.Parameter((n_s * n_x, K)),
            "D": cvx.Parameter((n_s * n_u, K)),
            "G": cvx.Parameter((n_s * n_p, K)),
            "r_path": cvx.Parameter((n_s, K)),
            # virtual control params
            "H0": cvx.Parameter((n_ic, n_x)),
            "K0": cvx.Parameter((n_ic, n_p)),
            "l0": cvx.Parameter(n_ic),
            "Hf": cvx.Parameter((n_tc, n_x)),
            "Kf": cvx.Parameter((n_tc, n_p)),
            "lf": cvx.Parameter(n_tc),
            # box constraints params
            "u_min": np.array(self.sp.F_limits[0]).reshape(-1, 1),
            "u_max": np.array(self.sp.F_limits[1]).reshape(-1, 1),
            # trust-region center and radius
            "X_ref": cvx.Parameter((n_x, K)),
            "U_ref": cvx.Parameter((n_u, K)),
            "p_ref": cvx.Parameter(n_p),
            "tr_eta": cvx.Parameter(nonneg=True),
            # dt_scale for effort cost (updated during convexification)
            "dt_scale": cvx.Parameter(nonneg=True),
            # Docking constraints parameters (orientation + centering)
            # "docking_slack": cvx.Parameter((2, K), nonneg=True),  # [psi_slack, centering_slack] per timestep
            # "centerline_perp": cvx.Parameter(2),  # perpendicular to centerline
            # "centerline_ref": cvx.Parameter(),  # reference value perp @ A
            # "docking_active": cvx.Parameter(K, nonneg=True),  # 1 if active, 0 if not
        }

        return problem_parameters

    def _get_constraints(self) -> list[cvx.Constraint]:
        """
        Define constraints for SCvx.
        """
        n_x, n_u, n_p = self.n_x, self.n_u, self.n_p
        n_s, n_ic, n_tc = self.n_s, self.n_ic, self.n_tc
        X = self.variables["X"]
        U = self.variables["U"]
        p = self.variables["p"]
        nu_dyn = self.variables["nu_dyn"]
        nu_s = self.variables["nu_s"]
        nu_ic = self.variables["nu_ic"]
        nu_tc = self.variables["nu_tc"]

        # Parameters
        A_bar = self.problem_parameters["A_bar"]
        B_plus_bar = self.problem_parameters["B_plus_bar"]
        B_minus_bar = self.problem_parameters["B_minus_bar"]
        F_bar = self.problem_parameters["F_bar"]
        r_bar = self.problem_parameters["r_bar"]

        C = self.problem_parameters["C"]
        D = self.problem_parameters["D"]
        G = self.problem_parameters["G"]
        r_path = self.problem_parameters["r_path"]

        H0 = self.problem_parameters["H0"]
        K0 = self.problem_parameters["K0"]
        l0 = self.problem_parameters["l0"]

        Hf = self.problem_parameters["Hf"]
        Kf = self.problem_parameters["Kf"]
        lf = self.problem_parameters["lf"]

        constr = []

        # (1) Linearized dynamics with ν_k
        for k in range(self.params.K - 1):
            constr += [
                X[:, k + 1]
                == cvx.reshape(A_bar[:, k], (n_x, n_x)) @ X[:, k]
                + cvx.reshape(B_minus_bar[:, k], (n_x, n_u)) @ U[:, k]
                + cvx.reshape(B_plus_bar[:, k], (n_x, n_u)) @ U[:, k + 1]
                + cvx.reshape(F_bar[:, k], (n_x, n_p)) @ p
                + r_bar[:, k]
                + nu_dyn[:, k]
            ]

        # (2) Path constraints
        if n_s > 0:
            for k in range(self.params.K):
                constr += [
                    cvx.reshape(C[:, k], (n_s, n_x), order="C") @ X[:, k]
                    + cvx.reshape(D[:, k], (n_s, n_u), order="C") @ U[:, k]
                    + cvx.reshape(G[:, k], (n_s, n_p), order="C") @ p
                    + r_path[:, k]
                    <= nu_s[:, k]
                ]
            # Non-negativity of path slacks
            constr += [nu_s >= 0]

        # (3) Initial constraint
        constr += [H0 @ X[:, 0] + K0 @ p + l0 + nu_ic == 0]

        # (4) Terminal constraint
        constr += [Hf @ X[:, -1] + Kf @ p + lf + nu_tc == 0]

        # (5) Input bounds
        u_min = self.problem_parameters["u_min"]
        u_max = self.problem_parameters["u_max"]
        constr += [U >= u_min, U <= u_max]

        # State bounds (extracted from LineString boundary in static obstacles)
        if self.state_bounds is not None:
            x_min = self.state_bounds["x_min"]
            x_max = self.state_bounds["x_max"]
            y_min = self.state_bounds["y_min"]
            y_max = self.state_bounds["y_max"]
            constr += [X[0, :] >= x_min, X[0, :] <= x_max]
            constr += [X[1, :] >= y_min, X[1, :] <= y_max]

        # (5.1) Initial and Final Input Constraints
        constr += [U[:, 0] == 0, U[:, -1] == 0]

        # (5.2) Maximum Time Constraint
        if self.n_p > 0:
            constr += [p[0] <= self.params.tf_max]
            constr += [p[0] >= 0.1]  # Minimum time constraint to avoid dt=0

        # (6) Trust region - Using infinity norm for intuitive box-like trust region
        eta = self.problem_parameters["tr_eta"]
        X_ref = self.problem_parameters["X_ref"]
        U_ref = self.problem_parameters["U_ref"]
        p_ref = self.problem_parameters["p_ref"]

        if self.n_p > 0:

            for k in range(self.params.K):
                constr += [
                    0.5 * cvx.norm(X[:, k] - X_ref[:, k], 1)
                    + 0.5 * cvx.norm(U[:, k] - U_ref[:, k], 1)
                    + 0.05 * cvx.norm(p - p_ref, 1)
                    <= eta,
                ]

        else:
            for k in range(self.params.K):
                constr += [
                    0.5 * cvx.norm(X[:, k] - X_ref[:, k], 1) + 0.5 * cvx.norm(U[:, k] - U_ref[:, k], 1) <= eta,
                ]

        # Constraint on final angular velocity
        omega_tol = 1e-3
        constr += [cvx.abs(X[5, -1]) <= omega_tol]
        # if isinstance(self.planning_goal, DockingTarget):
        #     docking_slack = self.problem_parameters["docking_slack"]
        #     goal_psi = self.goal_state.psi
        #     perp = self.problem_parameters["centerline_perp"]
        #     ref = self.problem_parameters["centerline_ref"]
        #     active = self.problem_parameters["docking_active"]

        #     nu_docking = self.variables["nu_docking"]

        #     for k in range(self.params.K):
        #         # Orientation: |psi - goal| <= slack_param + nu_docking[0, k]
        #         constr += [cvx.abs(X[2, k] - goal_psi) <= docking_slack[0, k] + nu_docking[0, k]]
        #         # Centering: |perp @ x - ref| <= slack_param + nu_docking[1, k] (when active)
        #         constr += [active[k] * cvx.abs(perp @ X[0:2, k] - ref) <= docking_slack[1, k] + nu_docking[1, k]]

        return constr

    def _get_objective(self) -> Union[cvx.Minimize, cvx.Maximize]:
        """
        Linear cost $L_{\lambda}(x, u, p, \hat{\nu})$ for the convex subproblem:
        $L_{\lambda} = \sum_k [ \Gamma_d(x_k, u_k, p) + \lambda P(\nu_k, \nu_{s,k}) ] + \Phi(x(1), p)$
        $+ \lambda(P(0, \nu_{ic}) + P(0, \nu_{tc}))$

        The structure mirrors _compute_J_lambda, except virtual controls ν replace
        nonlinear defects. Φ(x(1),p) ≡ 0 in this setup.
        """
        distance_terms = []
        effort_terms = []

        for k in range(self.params.K):
            u_k = self.variables["U"][:, k]
            effort_terms.append(cvx.sum_squares(u_k))

            if k != 0:
                delta_xy = self.variables["X"][0:2, k] - self.variables["X"][0:2, k - 1]
                distance_terms.append(cvx.norm(delta_xy, 2))

        # === Stage cost Γ_d(x_k,u_k,p) ===
        J_time = self.params.weight_time * self.variables["p"][0]
        J_distance = self.params.weight_distance * cvx.sum(distance_terms)
        # Use dt_scale parameter (updated during convexification from p_bar[0]/(K-1))
        # This is DCP-compliant since Parameter * convex_expr is convex
        J_effort = self.params.weight_effort * self.problem_parameters["dt_scale"] * cvx.sum(effort_terms)

        # === Penalty part λP(ν_k, ν_{s,k}) ===
        # Here ν_dyn and ν_s are the convex variables replacing nonlinear defects.
        nu_dyn = self.variables["nu_dyn"]
        nu_s = self.variables["nu_s"]
        nu_ic = self.variables["nu_ic"]
        nu_tc = self.variables["nu_tc"]

        J_nu_dyn = self.params.weight_nu_dyn * cvx.sum(cvx.abs(nu_dyn))
        J_nu_path = self.params.weight_nu_path * cvx.sum(nu_s)
        J_nu_boundary = self.params.weight_nu_boundary * (cvx.sum(cvx.abs(nu_ic)) + cvx.sum(cvx.abs(nu_tc)))

        # Docking penalty (orientation + centering slack)
        # nu_docking = self.variables["nu_docking"]
        # J_nu_docking = self.params.weight_nu_docking * cvx.sum(nu_docking)

        J_nu = J_nu_dyn + J_nu_path + J_nu_boundary  # + J_nu_docking

        # Φ(x(1),p) term is zero; all remaining penalties captured in J_nu.
        objective = J_time + J_distance + J_effort + J_nu

        return cvx.Minimize(objective)

    def _convexification(self):
        """
        Perform convexification step, i.e. Linearization and Discretization
        and populate Problem Parameters.
        """
        n_x, n_u, n_p, K = self.n_x, self.n_u, self.n_p, self.params.K

        # 1) Linearization + discretization around the current reference trajectory
        #    For FOH:
        #      x_{k+1} = A_bar_k x_k + B_minus_bar_k u_k + B_plus_bar_k u_{k+1} + F_bar_k p + r_bar_k
        #    For ZOH (if you ever switch): same idea but with a single B_bar_k.
        if isinstance(self.integrator, FirstOrderHold):
            A_bar, B_plus_bar, B_minus_bar, F_bar, r_bar = self.integrator.calculate_discretization(
                self.X_bar, self.U_bar, self.p_bar
            )
            self.problem_parameters["A_bar"].value = A_bar
            self.problem_parameters["B_plus_bar"].value = B_plus_bar
            self.problem_parameters["B_minus_bar"].value = B_minus_bar
            self.problem_parameters["F_bar"].value = F_bar
            self.problem_parameters["r_bar"].value = r_bar

        elif isinstance(self.integrator, ZeroOrderHold):
            A_bar, B_bar, F_bar, r_bar = self.integrator.calculate_discretization(self.X_bar, self.U_bar, self.p_bar)
            # You can reuse B_bar for both plus/minus or change constraints to only use B_bar
            self.problem_parameters["A_bar"].value = A_bar
            self.problem_parameters["B_plus_bar"].value = B_bar
            self.problem_parameters["B_minus_bar"].value = B_bar
            self.problem_parameters["F_bar"].value = F_bar
            self.problem_parameters["r_bar"].value = r_bar

        else:
            raise RuntimeError("Unknown discretization method in SatellitePlanner.")

        # 2) Initial and goal states (as vectors in R^{n_x})
        if hasattr(self, "init_state"):
            self.problem_parameters["init_state"].value = self._state_to_vec(self.init_state)

        if hasattr(self, "goal_state"):
            self.problem_parameters["goal_state"].value = self._state_to_vec(self.goal_state)

        # 3) Trust-region center and radius
        #    Center is the current reference trajectory (X_bar, U_bar, p_bar),
        #    radius comes from SolverParameters (and will be updated by _update_trust_region()).
        self.problem_parameters["X_ref"].value = self.X_bar
        self.problem_parameters["U_ref"].value = self.U_bar
        self.problem_parameters["p_ref"].value = self.p_bar
        self.problem_parameters["tr_eta"].value = self.tr_eta

        if self.n_p > 0:
            self.dt_scale = float(self.p_bar[0] / (self.params.K - 1))
        else:
            self.dt_scale = 1.0

        # Update dt_scale parameter for the convex problem
        self.problem_parameters["dt_scale"].value = self.dt_scale

        # Populate H0, K0, l0 for initial condition X[:,0] == init_state
        # H0 = I, K0 = 0, l0 = -init_state
        self.problem_parameters["H0"].value = np.eye(n_x)
        self.problem_parameters["K0"].value = np.zeros((n_x, n_p))
        self.problem_parameters["l0"].value = -self.problem_parameters["init_state"].value

        # Populate Hf, Kf, lf for terminal condition X[:,-1] == goal_state
        # Hf = I, Kf = 0, lf = -goal_state
        self.problem_parameters["Hf"].value = np.eye(n_x)
        self.problem_parameters["Kf"].value = np.zeros((n_x, n_p))
        self.problem_parameters["lf"].value = -self.problem_parameters["goal_state"].value

        # Populate C, D, G, r_path
        if self.n_s > 0:
            # Initialize to zeros
            C_val = np.zeros((self.n_s * n_x, K))
            D_val = np.zeros((self.n_s * n_u, K))
            G_val = np.zeros((self.n_s * n_p, K))
            r_path_val = np.zeros((self.n_s, K))

            sat_radius = self.sat_radius  # Approximation

            # Iterate over all obstacles
            # Order: Planets then Asteroids (must match self.n_s count logic if we had one,
            # but here we just iterate and increment index)

            obs_idx = 0

            # 1. Planets (Static)
            for planet in self.planets.values():
                p_center = np.array(planet.center)
                p_radius = planet.radius
                min_dist = p_radius + sat_radius * self.params.margin_collision_distance

                for k in range(K):
                    # Reference position
                    x_bar = self.X_bar[0:2, k]

                    # Vector from obstacle to satellite
                    diff = x_bar - p_center
                    dist = np.linalg.norm(diff)

                    if dist < 1e-6:
                        n_vec = np.array([1.0, 0.0])
                    else:
                        n_vec = diff / dist

                    base_idx = obs_idx * n_x
                    C_val[base_idx, k] = -n_vec[0]
                    C_val[base_idx + 1, k] = -n_vec[1]

                    r_path_val[obs_idx, k] = min_dist + np.dot(n_vec, p_center)

                obs_idx += 1

            # 2. Asteroids (Dynamic)
            for asteroid in self.asteroids.values():
                a_start = np.array(asteroid.start)

                # Rotate velocity based on orientation
                theta = asteroid.orientation
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])
                a_vel = R @ np.array(asteroid.velocity)

                a_radius = asteroid.radius
                min_dist = a_radius + sat_radius * self.params.margin_collision_distance

                for k in range(K):
                    # Estimated time for this step based on reference p_bar
                    # t_bar = k * dt_bar
                    # But we linearized w.r.t p.

                    # Reference position
                    x_bar = self.X_bar[0:2, k]

                    # Reference time and obstacle position
                    dt_bar = self.p_bar[0] / (K - 1) if self.n_p > 0 else 0.0
                    t_bar = k * dt_bar
                    a_pos_bar = a_start + a_vel * t_bar

                    diff = x_bar - a_pos_bar
                    dist = np.linalg.norm(diff)

                    if dist < 1e-6:
                        n_vec = np.array([1.0, 0.0])
                    else:
                        n_vec = diff / dist

                    # Linearized constraint:
                    # -n^T x + n^T v * (k/(K-1)) * p <= -R - n^T p_start

                    base_idx = obs_idx * n_x
                    C_val[base_idx, k] = -n_vec[0]
                    C_val[base_idx + 1, k] = -n_vec[1]

                    if self.n_p > 0:
                        # G is (n_s * n_p, K)
                        # We want row `obs_idx` of G_k to be [coeff]
                        # Flattened index: obs_idx * n_p + 0

                        coeff = np.dot(n_vec, a_vel) * (k / (K - 1))
                        G_val[obs_idx * self.n_p, k] = coeff

                    r_path_val[obs_idx, k] = min_dist + np.dot(n_vec, a_start)

                obs_idx += 1

            self.problem_parameters["C"].value = C_val
            self.problem_parameters["D"].value = D_val
            self.problem_parameters["G"].value = G_val
            self.problem_parameters["r_path"].value = r_path_val

    def _extract_state_bounds(self, static_obstacles: Sequence[StaticObstacle] | None) -> dict | None:
        """
        Extract state bounds from static obstacles.
        Finds the StaticObstacle with a LineString shape (the boundary)
        and extracts the min/max x and y coordinates.

        Args:
            static_obstacles: Sequence of static obstacles from dg_scenario

        Returns:
            Dictionary with x_min, x_max, y_min, y_max or None if no boundary found
        """
        if static_obstacles is None:
            return None

        for obstacle in static_obstacles:
            if isinstance(obstacle.shape, LineString):
                coords = np.array(obstacle.shape.coords)
                x_min = float(np.min(coords[:, 0]))
                x_max = float(np.max(coords[:, 0]))
                y_min = float(np.min(coords[:, 1]))
                y_max = float(np.max(coords[:, 1]))
                return {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                }

        return None

    def _state_to_vec(self, rs: Union[SatelliteState, DynObstacleState]) -> np.ndarray:
        return np.array([rs.x, rs.y, rs.psi, rs.vx, rs.vy, rs.dpsi])

    def _compute_J_lambda(self, X: NDArray, U: NDArray, p: NDArray) -> float:
        """
        Compute the nonlinear penalized cost J_λ for a given trajectory.
        This includes the base objective plus defects.
        "Replace the virtual control terms with the defects (for the dynamics constraints)
        and with the true nonconvex path and boundary constraint violations."

        $J_{\lambda}(x, u, p)$
            $= \sum_{k=1}^{K} (\Gamma_d(x_k, u_k, p)$
                $+ \lambda P(\delta_k, [s(t_k, x_k, u_k, p)]^+))$
                $+ \Phi(x(1), p)$
                $+ \lambda(P(0, g_{ic}(x(0), p))$
                $+ P(0, g_{tc}(x(1), p))$
            $)$

        Args:
            X: State trajectory (n_x, K)
            U: Control trajectory (n_u, K)
            p: Parameter vector (n_p,)

        Returns:
            Penalized cost J_λ
        """
        K = self.params.K

        # Use p_bar[0]/(K-1) to match the convex problem's dt_scale parameter
        # This ensures J_lambda and L_lambda use the same effort scaling
        dt = self.dt_scale  # = p_bar[0]/(K-1), updated in _convexification

        # === Stage cost Γ_d(x_k,u_k,p) ===
        # Components correspond to time, path length and thrust effort.
        J_time = self.params.weight_time * p[0] if self.satellite.n_p > 0 else 0.0

        deltas = [np.linalg.norm(X[0:2, k] - X[0:2, k - 1], 2) for k in range(1, K)]
        J_distance = self.params.weight_distance * float(np.sum(deltas))

        efforts = [np.sum(U[:, k] ** 2) for k in range(K)]
        J_effort = self.params.weight_effort * dt * float(np.sum(efforts))

        # === Penalty part λP(δ_k, [s]^+) ===
        # J_nu_dyn uses nonlinear dynamics defects δ_k instead of the virtual controls ν_dyn
        # that live in the convex subproblem, keeping Jλ faithful to the original dynamics.
        # J_nu_path corresponds to s(t_k,x_k,u_k,p) violations (obstacle constraints).
        J_nu_dyn = 0.0
        J_nu_path = 0.0
        J_nu_boundary = 0.0

        # 1. Dynamics defects δ_k
        # Integrate nonlinear dynamics piecewise to recover the "true" successor X_nl
        for k in range(K - 1):
            # Integrate one step from X[:, k] using U[:, k], U[:, k+1]
            x_next_nl = odeint(self.integrator._dxdt, X[:, k], self.integrator.range_t, args=(U[:, k], U[:, k + 1], p))[
                -1, :
            ]
            defect = np.linalg.norm(X[:, k + 1] - x_next_nl, 1)
            J_nu_dyn += defect

        # 2. Path constraints (Collision avoidance)
        sat_radius = self.sat_radius

        # Path violation buffer aggregates planet + asteroid distances per step
        for k in range(K):
            step_violations = []

            # Planets (static)
            for planet in self.planets.values():
                p_center = np.array(planet.center)
                p_radius = planet.radius
                min_dist = p_radius + sat_radius * self.params.margin_collision_distance

                sat_pos = X[0:2, k]
                dist = np.linalg.norm(sat_pos - p_center)
                step_violations.append(max(0.0, min_dist - dist))

            # Asteroids (dynamic)
            for asteroid in self.asteroids.values():
                a_start = np.array(asteroid.start)

                # Rotate velocity based on orientation
                theta = asteroid.orientation
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])
                a_vel = R @ np.array(asteroid.velocity)

                a_radius = asteroid.radius
                min_dist = a_radius + sat_radius * self.params.margin_collision_distance

                t = k * dt
                a_pos = a_start + a_vel * t
                sat_pos = X[0:2, k]
                dist = np.linalg.norm(sat_pos - a_pos)
                step_violations.append(max(0.0, min_dist - dist))

            if step_violations:
                J_nu_path += np.linalg.norm(step_violations, 1)

        # === Terminal terms ===
        # Φ(x(1),p) is zero in this setup; only equality-constraint penalties remain.
        # These enforce P(0, g_ic(x(0),p)) and P(0, g_tc(x(1),p)).
        # Initial state
        init_vec = self._state_to_vec(self.init_state)
        J_nu_boundary += np.linalg.norm(X[:, 0] - init_vec, 1)

        # Terminal state
        goal_vec = self._state_to_vec(self.goal_state)
        J_nu_boundary += np.linalg.norm(X[:, -1] - goal_vec, 1)

        # Apply weights
        J_nu_dyn *= self.params.weight_nu_dyn
        J_nu_path *= self.params.weight_nu_path
        J_nu_boundary *= self.params.weight_nu_boundary

        J_nu = J_nu_dyn + J_nu_path + J_nu_boundary

        total_cost = J_time + J_distance + J_effort + J_nu

        if PlannerConfig.VERBOSE:
            print(
                f"[J_lambda] time={J_time:.3f} dist={J_distance:.3f} effort={J_effort:.3f} "
                f"nu={J_nu:.3f} (dyn={J_nu_dyn:.3f}, path={J_nu_path:.3f}, bound={J_nu_boundary:.3f}) "
                f"total={total_cost:.3f}"
            )

        return total_cost

    def _check_convergence(self, X_new: NDArray, U_new: NDArray, p_new: NDArray, L_new: float, J_new: float) -> bool:
        """
        Check convergence of SCvx using one of three criteria.

        Args:
            X_new: New state trajectory (n_x, K)
            U_new: New control trajectory (n_u, K)
            p_new: New parameter vector (n_p,)
            L_new: Convexified cost L_λ at the new solution
            J_new: Penalized nonlinear cost J_λ at the new solution.

        Returns:
            True if converged according to selected criterion
        """
        mode = self.params.stop_crit_mode
        eps = self.params.stop_crit

        if mode == StopCritMode.ABSOLUTE_COST:
            # Criterion 1: $|J_{new} - L_{new}| \leq \epsilon$ (gap between nonlinear and convex cost at new solution)
            gap = abs(J_new - L_new)
            converged = gap <= eps
            if PlannerConfig.VERBOSE:
                print(
                    f"[SCvx][Conv] ABSOLUTE_COST: |J_new - L_new| = |{J_new:.3f} - {L_new:.3f}| = {gap:.4f} {'<=' if converged else '>'} {eps}"
                )
            return converged

        if mode == StopCritMode.STATE_PARAM:
            # Criterion 2: $||p^* - \bar{p}||_q + \max_k ||x_k^* - \bar{x}_k||_q \leq \epsilon$
            q = self.params.stop_crit_norm

            # Parameter difference
            dp = np.linalg.norm(p_new - self.p_bar, ord=q)

            # Maximum per-timestep state difference
            max_dx = 0.0
            for k in range(self.params.K):
                dx_k = np.linalg.norm(X_new[:, k] - self.X_bar[:, k], ord=q)
                max_dx = max(max_dx, dx_k)

            total_diff = dp + max_dx
            converged = total_diff <= eps
            if PlannerConfig.VERBOSE:
                print(
                    f"[SCvx][Conv] STATE_PARAM: dp={dp:.4f} + max_dx={max_dx:.4f} = {total_diff:.4f} {'<=' if converged else '>'} {eps}"
                )
            return converged

        if mode == StopCritMode.RELATIVE_COST:
            # Criterion 3: $(J_{new} - L_{new}) \leq \epsilon_r |J_{new}|$
            eps_r = self.params.stop_crit_rel
            gap = J_new - L_new
            threshold = eps_r * abs(J_new)
            converged = gap <= threshold
            if PlannerConfig.VERBOSE:
                print(
                    f"[SCvx][Conv] RELATIVE_COST: (J_new - L_new) = ({J_new:.3f} - {L_new:.3f}) = {gap:.4f} {'<=' if converged else '>'} {eps_r}*|{J_new:.3f}| = {threshold:.4f}"
                )
            return converged

        raise ValueError(f"Unknown stop criterion mode: {mode}")

    def _update_trust_region(self):
        """
        Update trust region radius according to SCvx trust-region rules.

        Parameters
        ----------
        J_old : float
            Penalized nonlinear cost J_λ at the old reference trajectory.
        J_new : float
            Penalized nonlinear cost J_λ at the new candidate trajectory
            (evaluated with true nonlinear dynamics).
        L_new : float
            Convexified penalized cost L_λ at the new candidate
            (i.e. the CVXPY problem objective value).

        Returns
        -------
        accept_step : bool
            Whether to accept the new trajectory as the next reference.
        """
        params = self.params

        # Predicted improvement (denominator) and actual improvement (numerator)
        num = self.J_old - self.J_new
        den = self.J_old - self.L_new

        # Guard against division by zero or negative predicted improvement
        if den <= 0.0:
            rho = 0.0
        else:
            rho = num / den

        eta_old = self.tr_eta
        eta = eta_old

        # Default: accept step, then possibly override below
        accept_step = True

        if rho < params.rho_0:
            # Very poor model -> reject step and shrink TR
            accept_step = False
            eta = max(params.min_tr_radius, eta * params.alpha)

        elif rho < params.rho_1:
            # Poor-ish model -> accept step but shrink TR
            eta = max(params.min_tr_radius, eta * params.alpha)

        elif rho < params.rho_2:
            # Good model -> accept, keep TR radius
            # eta unchanged
            pass

        else:
            # Very good / conservative model -> accept and enlarge TR
            eta = min(params.max_tr_radius, eta * params.beta)

        # Store new radius
        self.tr_eta = eta

        if PlannerConfig.VERBOSE:
            status = "accepted" if accept_step else "rejected"
            print(f"[SCvx][TR] rho={rho:.3f} -> {status}, radius {eta_old:.3f} -> {eta:.3f}")

        return accept_step

    def _extract_seq_from_array(
        self, X_new: np.ndarray, U_new: np.ndarray, p_new: np.ndarray
    ) -> tuple[DgSampledSequence[SatelliteCommands], DgSampledSequence[SatelliteState]]:
        """
        Convert the optimized state/control arrays and total time into DgSampledSequence objects for commands and states.
        """
        K = self.params.K
        total_time = float(p_new[0])
        ts = tuple(np.linspace(0.0, total_time, self.params.K))

        cmds = [SatelliteCommands(*U_new[:, k].tolist()) for k in range(K)]
        states = [SatelliteState(*X_new[:, k].tolist()) for k in range(K)]

        cmds_seq = DgSampledSequence[SatelliteCommands](timestamps=ts, values=cmds)
        states_seq = DgSampledSequence[SatelliteState](timestamps=ts, values=states)

        return cmds_seq, states_seq
