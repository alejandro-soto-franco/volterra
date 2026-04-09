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
