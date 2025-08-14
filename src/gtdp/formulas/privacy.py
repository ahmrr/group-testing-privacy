import math
import numpy as np

from ..utils import IntLike, FloatLike


def _epsilon(
    p: float,
    c: int,
    Se: float,
    Sp: float,
) -> float:
    """Differential privacy level (reference implementation)"""

    if p == 1:
        return 0
    elif p == 0 or c == 1:
        return math.log(Se / (1 - Sp) if Se < Sp else Sp / (1 - Se))
    else:
        r = Se + Sp - 1

        α = 1 + r * (1 - p) ** (c - 1) / (1 - Se)
        β = 1 - r * (1 - p) ** (c - 1) / Se
        crit = 1 - ((2 * Se - 1) / r) ** (1 / (c - 1))

        return -math.log(β) if Se < Sp and p < crit else math.log(α)


def epsilon(
    p: FloatLike,
    c: IntLike,
    Se: FloatLike,
    Sp: FloatLike,
) -> FloatLike:
    """Differential privacy level"""

    # Add singleton dimensions to broadcast properly
    p = np.atleast_1d(p)[:, None, None, None]
    c = np.atleast_1d(c)[None, :, None, None]
    Se = np.atleast_1d(Se)[None, None, :, None]
    Sp = np.atleast_1d(Sp)[None, None, None, :]

    p, c, Se, Sp = np.broadcast_arrays(p, c, Se, Sp)

    # Compute differentiala privacy epsilon according to formula
    r = Se + Sp - 1
    α = 1 + r * (1 - p) ** (c - 1) / (1 - Se)
    β = 1 - r * (1 - p) ** (c - 1) / Se
    p_crit = 1 - ((2 * Se - 1) / r) ** (1 / (c - 1))
    res = np.select(
        condlist=[
            np.isclose(p, 1),
            np.isclose(p, 0) | (c == 1),
            (Se < Sp) & (p < p_crit),
            (Se >= Sp) | (p >= p_crit),
        ],
        choicelist=[
            np.zeros_like(p),
            np.where(Se < Sp, Se / (1 - Sp), Sp / (1 - Se)),
            -np.log(β),
            np.log(α),
        ],
    ).squeeze()

    # Return float if only a single value is present
    return float(res) if res.size == 1 else res


def _add_noise(
    Se: float,
    Sp: float,
    Γ1: float,
    Γ2: float,
) -> tuple[float, float]:
    """Effective Se and Sp after adding noise (reference implementation)."""

    ρ = Γ1 + Γ2 - 1

    assert (Γ2 - 0.5) / ρ < Se < Γ2 / ρ and (Γ1 - 0.5) / ρ < Sp < Γ1 / ρ

    return (
        1 - Γ2 + ρ * Se,
        1 - Γ1 + ρ * Sp,
    )


def add_noise(
    Se: FloatLike,
    Sp: FloatLike,
    Γ1: FloatLike,
    Γ2: FloatLike,
) -> tuple[FloatLike, FloatLike]:
    """Effective sensitivity and specificity after adding noise"""

    # Add singleton dimensions to broadcast properly
    Se = np.atleast_1d(Se)[:, None, None, None]
    Sp = np.atleast_1d(Sp)[None, :, None, None]
    Γ1 = np.atleast_1d(Γ1)[None, None, :, None]
    Γ2 = np.atleast_1d(Γ2)[None, None, None, :]

    # TODO
    ρ = Γ1 + Γ2 - 1

    # assert (Γ2 - 0.5) / ρ < Se < Γ2 / ρ and (Γ1 - 0.5) / ρ < Sp < Γ1 / ρ

    return (
        1 - Γ2 + ρ * Se,
        1 - Γ1 + ρ * Sp,
    )
