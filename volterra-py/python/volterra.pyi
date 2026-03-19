"""
volterra.pyi -- type stubs for the volterra Python extension module.
Generated manually; update when the Rust API changes.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# MarsParams
# ---------------------------------------------------------------------------

class MarsParams:
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
    ) -> None: ...

    @staticmethod
    def default_test() -> MarsParams: ...

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

def run_mars_component1(
    q_init: QField2D,
    params: MarsParams,
    n_steps: int,
    snap_every: int,
) -> tuple[QField2D, list[SnapStats]]:
    """
    Run Component 1: single-phase MARS dry active nematic.

    Parameters
    ----------
    q_init : QField2D
        Initial Q-tensor field.
    params : MarsParams
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

def k0_convolution(q_rot: QField2D, params: MarsParams) -> QField2D:
    """
    Apply the K₀ transfer map ℳ_SM(Q_rot).

    Parameters
    ----------
    q_rot : QField2D
        Rotor field (Component 1 output).
    params : MarsParams
        Uses params.xi_l for the kernel width.

    Returns
    -------
    QField2D
        Driven lipid Q-field.
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
