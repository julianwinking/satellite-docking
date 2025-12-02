import sympy as spy

from dg_commons.sim.models.satellite_structures import SatelliteGeometry, SatelliteParameters


class SatelliteDyn:
    sg: SatelliteGeometry
    sp: SatelliteParameters

    x: spy.Matrix
    u: spy.Matrix
    p: spy.Matrix

    n_x: int
    n_u: int
    n_p: int

    f: spy.Function
    A: spy.Function
    B: spy.Function
    F: spy.Function

    def __init__(self, sg: SatelliteGeometry, sp: SatelliteParameters):
        self.sg = sg
        self.sp = sp

        self.x = spy.Matrix(spy.symbols("x y psi vx vy dpsi", real=True))  # states
        self.u = spy.Matrix(spy.symbols("thrust_l thrust_r", real=True))  # inputs
        self.p = spy.Matrix([spy.symbols("t_f", positive=True)])  # final time

        self.n_x = self.x.shape[0]  # number of states
        self.n_u = self.u.shape[0]  # number of inputs
        self.n_p = self.p.shape[0]

    def get_dynamics(self) -> tuple[spy.Function, spy.Function, spy.Function, spy.Function]:
        """
        Define dynamics and extract jacobians.
        Get dynamics for SCvx.
        extract the state from self.x the following way:
        0x 1y 2psi 3vx 4vy 5dpsi
        """
        # HINT for the dynamics: SymPy is a library for symbolic mathematics.
        # Here you’ll need symbolic math, not numerical math.
        # That means you should use SymPy functions (like sympy.sin, sympy.exp, …)
        # together with symbols, instead of numerical functions from numpy or math.

        # physical parameters
        m = spy.Float(self.sp.m_v)
        I = spy.Float(self.sg.Iz)
        l_m = spy.Float(self.sg.l_m)

        # dynamics in normalized time tau
        # (tau in [0, 1]); p = t_f scales derivatives
        thrust_total = self.u[0] + self.u[1]
        p_tf = self.p[0]

        f = spy.zeros(self.n_x, 1)
        f[0] = self.x[3] * p_tf
        f[1] = self.x[4] * p_tf
        f[2] = self.x[5] * p_tf
        f[3] = (spy.cos(self.x[2]) * thrust_total / m) * p_tf
        f[4] = (spy.sin(self.x[2]) * thrust_total / m) * p_tf
        f[5] = (l_m / I * (self.u[1] - self.u[0])) * p_tf

        # Jacobians and matrices of the system (don't need to change)
        A = f.jacobian(self.x)
        B = f.jacobian(self.u)
        F = f.jacobian(self.p)

        f_func = spy.lambdify((self.x, self.u, self.p), f, "numpy")
        A_func = spy.lambdify((self.x, self.u, self.p), A, "numpy")
        B_func = spy.lambdify((self.x, self.u, self.p), B, "numpy")
        F_func = spy.lambdify((self.x, self.u, self.p), F, "numpy")

        return f_func, A_func, B_func, F_func
