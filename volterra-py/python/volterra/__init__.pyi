"""
volterra.pyi -- type stubs for the volterra Python extension module.
Generated manually; update when the Rust API changes.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# ActiveNematicParams
# ---------------------------------------------------------------------------

class ActiveNematicParams:
    nx: int
    ny: int
    dx: float
    dt: float
    k_r: float
    gamma_r: float
    zeta_eff: float
    eta: float
    a_landau: float
    c_landau: float
    lambda_: float
    noise_amp: float
    k_l: float
    gamma_l: float
    xi_l: float
    chi_ms: float
    kappa_ch: float
    a_ch: float
    b_ch: float
    m_l: float
    zeta_field: float

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dt: float,
        k_r: float,
        gamma_r: float,
        zeta_eff: float,
        eta: float,
        a_landau: float,
        c_landau: float,
        lambda_: float,
        k_l: float,
        gamma_l: float,
        xi_l: float,
        noise_amp: float = 0.0,
        chi_ms: float = 0.5,
        kappa_ch: float = 1.0,
        a_ch: float = 1.0,
        b_ch: float = 1.0,
        m_l: float = 0.1,
    ) -> None: ...

    @staticmethod
    def default_test() -> ActiveNematicParams: ...

    def defect_length(self) -> float:
        """ℓ_d = sqrt(K_r / ζ_eff)"""
        ...

    def pi_number(self) -> float:
        """Π = K_r / (Γ_l η K_l) -- must be < 1 for coherent window."""
        ...

    def a_eff(self) -> float:
        """a_eff = a_landau - ζ_eff / 2."""
        ...

    def ch_coherence_length(self) -> float:
        """xi_CH = sqrt(kappa_ch / a_ch)."""
        ...

    def phi_eq(self) -> float:
        """phi_eq = sqrt(a_ch / b_ch)."""
        ...

    def validate(self) -> None:
        """Raises ValueError if any parameter is physically unreasonable."""
        ...

# ---------------------------------------------------------------------------
# QField2D
# ---------------------------------------------------------------------------

class QField2D:
    nx: int
    ny: int
    dx: float

    @staticmethod
    def zeros(nx: int, ny: int, dx: float) -> QField2D: ...

    @staticmethod
    def uniform(nx: int, ny: int, dx: float, q1: float, q2: float) -> QField2D: ...

    @staticmethod
    def random_perturbation(
        nx: int, ny: int, dx: float, amplitude: float, seed: int
    ) -> QField2D: ...

    @staticmethod
    def from_numpy(
        arr: npt.NDArray[np.float64],
        nx: int,
        ny: int,
        dx: float,
    ) -> QField2D:
        """Import from a (nx*ny, 2) float64 array."""
        ...

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Export as (nx*ny, 2) float64 array. Reshape to (nx, ny, 2) in Python."""
        ...

    def order_param(self) -> npt.NDArray[np.float64]:
        """S = 2*sqrt(q1^2 + q2^2) at each vertex, shape (nx*ny,)."""
        ...

    def director_angle(self) -> npt.NDArray[np.float64]:
        """theta = atan2(q2, q1)/2 in [-pi/2, pi/2], shape (nx*ny,)."""
        ...

    def mean_order_param(self) -> float: ...
    def max_norm(self) -> float: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# ScalarField2D
# ---------------------------------------------------------------------------

class ScalarField2D:
    nx: int
    ny: int
    dx: float

    @staticmethod
    def zeros(nx: int, ny: int, dx: float) -> ScalarField2D: ...

    @staticmethod
    def uniform(nx: int, ny: int, dx: float, val: float) -> ScalarField2D: ...

    @staticmethod
    def from_numpy(
        arr: npt.NDArray[np.float64], nx: int, ny: int, dx: float
    ) -> ScalarField2D:
        """Import from a (nx*ny,) float64 array."""
        ...

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Export as a (nx*ny,) float64 array."""
        ...

    def mean_value(self) -> float: ...
    def variance(self) -> float: ...
    def max_value(self) -> float: ...
    def min_value(self) -> float: ...
    def mean_gradient_sq(self) -> float: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

# ---------------------------------------------------------------------------
# VelocityField2D
# ---------------------------------------------------------------------------

class VelocityField2D:
    nx: int
    ny: int
    dx: float

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Export as (nx*ny, 2) float64 array. Reshape to (nx, ny, 2) in Python."""
        ...

# ---------------------------------------------------------------------------
# SnapStats
# ---------------------------------------------------------------------------

class SnapStats:
    time: float
    mean_s: float
    n_defects: int
    n_plus: int
    n_minus: int
    defect_density: float

# ---------------------------------------------------------------------------
# DefectInfo
# ---------------------------------------------------------------------------

class DefectInfo:
    plaquette: tuple[int, int]
    angle: float
    charge_sign: int   # +1 or -1

# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def run_dry_active_nematic(
    q_init: QField2D,
    params: ActiveNematicParams,
    n_steps: int,
    snap_every: int,
) -> tuple[QField2D, list[SnapStats]]:
    """
    Run the dry active nematic simulation.

    Parameters
    ----------
    q_init : QField2D
        Initial Q-tensor field.
    params : ActiveNematicParams
        All physical and numerical parameters.
    n_steps : int
        Total number of time steps.
    snap_every : int
        Record a snapshot every this many steps.

    Returns
    -------
    (QField2D, list[SnapStats])
        Final field and list of snapshots.
    """
    ...

def k0_convolution(q_rot: QField2D, params: ActiveNematicParams) -> QField2D:
    """
    Apply the K₀ transfer map ℳ_SM(Q_rot).

    Parameters
    ----------
    q_rot : QField2D
        Rotor field (Component 1 output).
    params : ActiveNematicParams
        Uses params.xi_l for the kernel width.

    Returns
    -------
    QField2D
        Driven lipid Q-field.
    """
    ...

def run_active_nematic_hydro(
    q_init: QField2D,
    params: ActiveNematicParams,
    n_steps: int,
    snap_every: int,
) -> tuple[QField2D, list[SnapStats]]:
    """
    Run Component 1 with full hydrodynamic flow coupling (spectral Stokes solver).

    At each step the Stokes velocity field is re-solved from the active stress
    sigma^a = zeta_eff * Q, enabling the active flow instability and turbulence.

    Returns (QField2D, list[SnapStats]).
    """
    ...

def stokes_solve(q: QField2D, params: ActiveNematicParams) -> VelocityField2D:
    """
    Solve the 2D incompressible Stokes equation for the active velocity.

    Returns the velocity field driven by sigma^a = zeta_eff * Q via spectral
    inversion of the stream-function biharmonic equation.
    """
    ...

def scan_defects(
    q: QField2D,
    threshold: float = 1.5707963267948966,  # pi/2
) -> list[DefectInfo]:
    """
    Holonomy-based defect detection.

    Returns a list of DefectInfo for each detected ±1/2 disclination.
    """
    ...

# ---------------------------------------------------------------------------
# Braid-group analysis (volterra-braid)
# ---------------------------------------------------------------------------

class BraidWord:
    """A braid word in the Artin generators (+i = sigma_i, -i = sigma_i^-1)."""

    n_strands: int
    codes: list[int]

    def __init__(self, n_strands: int, codes: Sequence[int]) -> None: ...

    @staticmethod
    def from_frames(
        frames: Sequence[Sequence[tuple[float, float, int]]],
    ) -> BraidWord:
        """Track a defect-position time series (frames of (x, y, charge)) and
        extract its braid word."""
        ...

    def entropy(self) -> float:
        """Topological entropy: log of the dilatation (Burau at t = -1)."""
        ...

    def permutation(self) -> list[int]:
        """perm[i] = final position of the strand that started at position i."""
        ...

    def exponent_sum(self) -> int: ...
    def fundamental_period(self) -> list[int]: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

def braid_detect_defects(
    qxx: Sequence[float],
    qxy: Sequence[float],
    nx: int,
    ny: int,
    threshold: float,
    mask: Sequence[bool],
) -> list[tuple[float, float, int]]:
    """Detect defects by the saddle-splay density, thresholded.

    `threshold` bounds the saddle-splay quantity, not an angle. Its scale
    follows the field's gradients; the reference draws at 0.05 * S0, about 0.07.
    Prefer braid_detect_defects_winding, which needs no threshold.
    """
    ...


def braid_detect_defects_winding(
    qxx: Sequence[float],
    qxy: Sequence[float],
    nx: int,
    ny: int,
    mask: Sequence[bool],
) -> list[tuple[float, float, int]]:
    """Detect defects by the director's holonomy. No threshold to choose."""
    ...

def braid_word_from_frames(
    frames: Sequence[Sequence[tuple[float, float, int]]],
) -> tuple[int, list[int]]:
    """Track + extract; returns (n_strands, codes)."""
    ...

def braid_topological_entropy(n_strands: int, codes: Sequence[int]) -> float:
    """Topological entropy of the braid given by (n_strands, codes)."""
    ...

# ---------------------------------------------------------------------------
# Confined boundary geometry (volterra-dec)
# ---------------------------------------------------------------------------

class PlaneCurve:
    """A closed plane curve: the wall a confined run is meshed against."""

    @property
    def period(self) -> float:
        """Parameter length of one circuit."""
        ...

    @property
    def features(self) -> list[float]:
        """Parameters of the corners and cusps that set the local element size."""
        ...

    @staticmethod
    def epitrochoid(q: float, d: float = 0.99, r: float = 1.0) -> PlaneCurve:
        """Regularised epitrochoid with 2(q - 1) cusps at outer scale r.

        d = 1 is the epicycloid with true cusps; d = 0 is a circle.
        """
        ...

    @staticmethod
    def from_points(
        points: npt.ArrayLike,
        features: Sequence[float] | None = None,
    ) -> PlaneCurve:
        """Closed table of (n, 2) points, splined. The parameter is the row index."""
        ...

    @staticmethod
    def from_callable(
        f: object,
        samples: int = 1024,
        period: float = 6.283185307179586,
        features: Sequence[float] | None = None,
    ) -> PlaneCurve:
        """Parametrisation f(u) -> (x, y) over [0, period), sampled and splined."""
        ...

    def point(self, u: float) -> tuple[float, float]: ...
    def tangent(self, u: float) -> tuple[float, float]: ...
    def inward_normal(self, u: float) -> tuple[float, float]: ...
    def curvature_radius(self, u: float) -> float: ...
    def speed(self, u: float) -> float: ...
    def sample(self, n: int) -> npt.NDArray[np.float64]:
        """n points spread uniformly in the parameter, shape (n, 2)."""
        ...

class ConfinedMesh:
    """Boundary-conforming graded mesh of a PlaneCurve's interior."""

    @property
    def vertices(self) -> npt.NDArray[np.float64]: ...
    @property
    def triangles(self) -> npt.NDArray[np.int64]: ...
    @property
    def boundary_vertices(self) -> npt.NDArray[np.int64]: ...
    @property
    def boundary_params(self) -> npt.NDArray[np.float64]: ...
    @property
    def boundary_normals(self) -> npt.NDArray[np.float64]: ...
    @property
    def n_vertices(self) -> int: ...
    @property
    def n_triangles(self) -> int: ...
    @property
    def n_boundary(self) -> int: ...
    @property
    def min_angle_deg(self) -> float: ...
    @property
    def max_angle_deg(self) -> float: ...
    @property
    def obtuse(self) -> int: ...
    @property
    def worst_cot_weight(self) -> float: ...
    @property
    def min_edge(self) -> float: ...
    @property
    def max_edge(self) -> float: ...
    @property
    def min_area(self) -> float: ...

    def imposed_charge(self, q_anchor: float = 1.0) -> tuple[float, float, int]:
        """(charge, worst step in degrees, steps over 90 degrees)."""
        ...

    @property
    def boundary_normal_angles(self) -> npt.NDArray[np.float64]:
        """Angle of the outward normal at each boundary vertex."""
        ...

    def anchoring_q(
        self, q_anchor: float = 1.0, s0: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Dirichlet (Qxx, Qxy) at each boundary vertex, shape (n_boundary, 2)."""
        ...

    def anchoring_director(self, q_anchor: float = 1.0) -> npt.NDArray[np.float64]:
        """Anchored director at each boundary vertex, shape (n_boundary, 2)."""
        ...

def confined_mesh(
    curve: PlaneCurve,
    h_bulk: float = 1.0,
    h_min: float = 0.05,
    grade: float = 1.3,
    boundary_frac: float = 0.25,
    smooth_passes: int = 8,
    seed: int = 0,
    cusp_edge: float = 0.0,
) -> ConfinedMesh:
    """Build a boundary-conforming graded mesh of the curve's interior."""
    ...

# ---------------------------------------------------------------------------
# BECH (Beris-Edwards-Cahn-Hilliard), 2D
# ---------------------------------------------------------------------------

class BechStats:
    time: float
    mean_s: float
    mean_phi: float
    phi_variance: float
    mean_grad_phi_sq: float
    n_defects: int
    n_plus: int
    n_minus: int
    defect_density: float

def run_bech(
    q_init: QField2D,
    phi_init: ScalarField2D,
    params: ActiveNematicParams,
    n_steps: int,
    snap_every: int,
) -> tuple[QField2D, ScalarField2D, list[BechStats]]:
    """Run the full Beris-Edwards-Cahn-Hilliard simulation."""
    ...

def ch_step_etd(
    phi: ScalarField2D,
    q_lip: QField2D,
    v: VelocityField2D,
    params: ActiveNematicParams,
) -> ScalarField2D:
    """Advance the Cahn-Hilliard field by one ETD1 step."""
    ...

# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

class ActiveNematicParams3D:
    nx: int
    ny: int
    nz: int
    dx: float
    dt: float
    k_r: float
    gamma_r: float
    zeta_eff: float
    eta: float
    a_landau: float
    c_landau: float
    b_landau: float
    lambda_: float
    noise_amp: float
    chi_a: float
    b0: float
    omega_b: float
    k_l: float
    gamma_l: float
    xi_l: float
    chi_ms: float
    kappa_ch: float
    a_ch: float
    b_ch: float
    m_l: float

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dt: float,
        k_r: float,
        gamma_r: float,
        zeta_eff: float,
        eta: float,
        a_landau: float,
        c_landau: float,
        b_landau: float,
        lambda_: float,
        k_l: float,
        gamma_l: float,
        xi_l: float,
        chi_ms: float,
        kappa_ch: float,
        a_ch: float,
        b_ch: float,
        m_l: float,
        chi_a: float = 0.0,
        b0: float = 1.0,
        omega_b: float = 1.0,
        noise_amp: float = 0.0,
        epsilon_a: float | None = None,
        e0: float | None = None,
        omega_e: float | None = None,
    ) -> None: ...

    @staticmethod
    def default_test() -> ActiveNematicParams3D: ...

    def defect_length(self) -> float:
        """ld = sqrt(K_r / zeta_eff)."""
        ...

    def pi_number(self) -> float:
        """Pi = K_r / (Gamma_l eta K_l)."""
        ...

    def a_eff(self) -> float:
        """a_eff = a_landau - zeta_eff / 2."""
        ...

    def ch_coherence_length(self) -> float:
        """xi_CH = sqrt(kappa_ch / a_ch)."""
        ...

    def phi_eq(self) -> float:
        """phi_eq = sqrt(a_ch / b_ch)."""
        ...

    def validate(self) -> None: ...
    def __repr__(self) -> str: ...

class QField3D:
    nx: int
    ny: int
    nz: int
    dx: float

    @property
    def q(self) -> npt.NDArray[np.float64]:
        """Q-tensor components, shape (nx, ny, nz, 5)."""
        ...

    @staticmethod
    def zeros(nx: int, ny: int, nz: int, dx: float) -> QField3D: ...

    @staticmethod
    def uniform(
        nx: int, ny: int, nz: int, dx: float,
        q11: float, q12: float, q13: float, q22: float, q23: float,
    ) -> QField3D:
        """The five independent components; q33 = -(q11 + q22)."""
        ...

    @staticmethod
    def random_perturbation(
        nx: int, ny: int, nz: int, dx: float, amplitude: float, seed: int
    ) -> QField3D: ...

    @staticmethod
    def from_numpy(
        arr: npt.NDArray[np.float64], nx: int, ny: int, nz: int, dx: float
    ) -> QField3D:
        """Import from a (nx*ny*nz, 5) float64 array."""
        ...

    def scalar_order(self) -> npt.NDArray[np.float64]:
        """S at each vertex, shape (nx*ny*nz,)."""
        ...

    def biaxiality(self) -> npt.NDArray[np.float64]:
        """P = lambda_mid - lambda_min at each vertex, shape (nx*ny*nz,)."""
        ...

    def mean_s(self) -> float: ...
    def max_norm(self) -> float: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class ScalarField3D:
    nx: int
    ny: int
    nz: int
    dx: float

    @property
    def phi(self) -> npt.NDArray[np.float64]:
        """Scalar values, flat, length nx*ny*nz."""
        ...

    @staticmethod
    def zeros(nx: int, ny: int, nz: int, dx: float) -> ScalarField3D: ...

    @staticmethod
    def uniform(nx: int, ny: int, nz: int, dx: float, val: float) -> ScalarField3D: ...

    @staticmethod
    def from_numpy(
        arr: npt.NDArray[np.float64], nx: int, ny: int, nz: int, dx: float
    ) -> ScalarField3D: ...

    def mean(self) -> float: ...
    def max(self) -> float: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

class VelocityField3D:
    nx: int
    ny: int
    nz: int
    dx: float

    @property
    def u(self) -> npt.NDArray[np.float64]:
        """Velocity components, shape (nx, ny, nz, 3)."""
        ...

    @staticmethod
    def zeros(nx: int, ny: int, nz: int, dx: float) -> VelocityField3D: ...

    @staticmethod
    def uniform(
        nx: int, ny: int, nz: int, dx: float, u: Sequence[float]
    ) -> VelocityField3D: ...

class SnapStats3D:
    time: float
    mean_s: float
    biaxiality_p: float
    n_disclination_lines: int
    total_line_length: float
    mean_line_curvature: float
    n_events: int

class BechStats3D:
    time: float
    mean_s: float
    mean_phi: float
    biaxiality_p: float
    n_disclination_lines: int
    total_line_length: float
    mean_line_curvature: float
    n_events: int

class DisclinationLine:
    @property
    def vertices(self) -> npt.NDArray[np.float64]:
        """Ordered vertex positions, shape (n, 3)."""
        ...

    @property
    def tangents(self) -> npt.NDArray[np.float64]:
        """Unit tangents at each vertex, shape (n, 3)."""
        ...

    @property
    def curvatures(self) -> npt.NDArray[np.float64]:
        """Frenet curvature at each vertex, shape (n,)."""
        ...

    @property
    def torsions(self) -> npt.NDArray[np.float64]:
        """Frenet-Serret torsion at each vertex, shape (n,)."""
        ...

    @property
    def charge(self) -> str:
        """"half_plus", "half_minus", or "anti"."""
        ...

    @property
    def is_loop(self) -> bool: ...

    def length(self) -> float:
        """Total arc length."""
        ...

    def mean_curvature(self) -> float: ...

class DisclinationEvent:
    @property
    def kind(self) -> str:
        """"creation", "annihilation", or "reconnection"."""
        ...

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """Approximate position, shape (3,)."""
        ...

    @property
    def frame(self) -> int: ...

def run_dry_active_nematic_3d(
    q_init: QField3D,
    params: ActiveNematicParams3D,
    n_steps: int,
    snap_every: int,
    out_dir: str,
    track_defects: bool,
) -> tuple[QField3D, list[SnapStats3D]]:
    """Run the dry 3D active nematic model."""
    ...

def run_bech_3d(
    q_init: QField3D,
    phi_init: ScalarField3D,
    params: ActiveNematicParams3D,
    n_steps: int,
    snap_every: int,
    out_dir: str,
    track_defects: bool,
) -> tuple[QField3D, ScalarField3D, list[BechStats3D]]:
    """Run the full 3D BECH model: Beris-Edwards, Stokes and Cahn-Hilliard."""
    ...

# ---------------------------------------------------------------------------
# Confined active nematic run
# ---------------------------------------------------------------------------

class ConfinedRun:
    """Beris-Edwards on a conforming mesh, stepped from Python."""

    def __init__(
        self,
        mesh: ConfinedMesh,
        active_length: float,
        coherence_length: float,
        resolution: int,
        q_anchor: float = 1.0,
        wall: str = "noslip",
        wall_h: float = 0.05,
        dt: float | None = None,
        seed: int = 0,
        picard: int = 1,
        full_stress: bool = True,
        elastic_mask: bool = True,
        elastic_h: float = 0.5,
        cg_tol: float = 1e-8,
        stokes_tol: float = 1e-8,
    ) -> None: ...

    def step(self, n: int = 1) -> None:
        """Advance by n steps. Raises RuntimeError if the field runs away."""
        ...

    def relax(self, n: int, tol: float = 1e-10) -> tuple[int, float]:
        """Passive relaxation with the flow off. Returns (steps, last change)."""
        ...

    def reset(self, seed: int = 0) -> None:
        """Draw a fresh random field and reset the clock."""
        ...

    @property
    def q(self) -> npt.NDArray[np.float64]:
        """(Qxx, Qxy) at every vertex, shape (n_vertices, 2)."""
        ...

    def set_q(self, q: npt.ArrayLike) -> None:
        """Replace the field, then re-impose the anchoring."""
        ...

    @property
    def velocity(self) -> npt.NDArray[np.float64]:
        """Velocity at every vertex, shape (n_vertices, 2)."""
        ...

    @property
    def order_parameter(self) -> npt.NDArray[np.float64]:
        """S = sqrt(Tr Q^2) at every vertex."""
        ...

    @property
    def director_angle(self) -> npt.NDArray[np.float64]:
        """Director angle in [-pi/2, pi/2) at every vertex."""
        ...

    @property
    def local_element_size(self) -> npt.NDArray[np.float64]: ...

    def defects(self, merge: float | None = None) -> list[tuple[float, float, int]]:
        """Detected disclinations as (x, y, charge), charge in half units."""
        ...

    def stats(self) -> dict[str, float]:
        """Per-step diagnostics. The counts are integers, the rest floats."""
        ...

    def free_energy(self) -> float: ...

    def params(self) -> dict[str, float]:
        """The dimensional constants."""
        ...

    @property
    def diffusive_dt_limit(self) -> float: ...
    @property
    def mesh(self) -> ConfinedMesh: ...
    @property
    def time(self) -> float: ...
    @property
    def n_steps(self) -> int: ...
    @property
    def dt(self) -> float: ...
    @property
    def wall(self) -> str:
        """"noslip" or "freeslip"."""
        ...
    @property
    def wall_vertices(self) -> int: ...
    @property
    def elastic_mask_vertices(self) -> int: ...
