import math
import numpy as np

from . import variance
from ..utils import IntLike, FloatLike

# utility = lambda p, c, J, Se, Sp: 1 / variance(p, c, J, Se, Sp)
# privacy = lambda p, c, Se, Sp: 1 / sp_epsilon(p, c, Se, Sp)

# Optimal (Se, Sp) results


def _optimal_accuracy(
    p: float,
    c: int,
    ε: float,
) -> tuple[float, float]:
    """Lowest variance Se and Sp that maintain ε-DP (reference implementation)"""

    λ1 = 1 + np.expm1(ε) / (1 - p) ** (c - 1)
    λ2 = 1 + np.expm1(-ε) / (1 - p) ** (c - 1)

    if ε == 0:
        return (0.5, 0.5)

    if λ2 > 0:
        S_crit = (λ1 - 1) / (λ1 - λ2)
        return (S_crit, λ1 * (1 - S_crit))
    else:
        S_min = 1 - 1 / λ1
        return (S_min, 1)


def optimal_accuracy(
    p: FloatLike,
    c: IntLike,
    ε: FloatLike,
) -> tuple[FloatLike, FloatLike]:
    """Lowest variance Se and Sp that maintain ε-DP"""

    # Add singleton dimensions to broadcast properly
    p = np.atleast_1d(p)[..., None, None]
    c = np.atleast_1d(c)[None, ..., None]
    ε = np.atleast_1d(ε)[None, None, ...]

    λ1 = 1 + np.expm1(ε) / (1 - p) ** (c - 1)
    λ2 = 1 + np.expm1(-ε) / (1 - p) ** (c - 1)
    cond_g = (λ2 > 0) & (ε != 0)
    cond_l = (λ2 <= 0) & (ε != 0)

    S_crit = (λ1 - 1) / (λ1 - λ2)
    S_min = 1 - 1 / λ1

    Se = np.select(
        condlist=[cond_g, cond_l],
        choicelist=[S_crit, S_min],
        default=0.5,
    ).squeeze()
    Sp = np.select(
        condlist=[cond_g, cond_l],
        choicelist=[λ1 * (1 - S_crit), 1],
        default=0.5,
    ).squeeze()

    if Se.size == 1:
        Se = float(Se)
    if Sp.size == 1:
        Sp = float(Sp)

    return Se, Sp


def _optimal_pool_size_variance(
    p: float,
    c: int,
    J: int,
    ε: float,
) -> float:
    """Variance of each pool size, considering optimal Se and Sp (reference implementation)"""

    c_crit = 1 + math.log1p(-math.exp(-ε)) / math.log1p(-p)

    if c > c_crit:
        return (
            (1 / ((1 - p) ** (c - 1)) - (1 - p))
            * (1 / math.expm1(ε) + (1 - p))
            / (J * c**2)
        )
    else:
        return (
            (1 / -math.expm1(-ε) - (1 - p)) * (1 / math.expm1(ε) + (1 - p)) / (J * c**2)
        )


def optimal_pool_size_variance(
    p: FloatLike,
    c: IntLike,
    J: IntLike,
    ε: FloatLike,
) -> FloatLike:
    """Variance of each pool size, considering optimal Se and Sp"""

    # Add singleton dimensions to broadcast properly
    p = np.atleast_1d(p)[..., None, None, None]
    c = np.atleast_1d(c)[None, ..., None, None]
    J = np.atleast_1d(J)[None, None, ..., None]
    ε = np.atleast_1d(ε)[None, None, None, ...]

    c_crit = 1 + np.log1p(-np.exp(-ε)) / np.log1p(-p)

    cond_g = c > c_crit
    cond_l = c <= c_crit

    res = np.select(
        condlist=[cond_g, cond_l],
        choicelist=[
            (
                (1 / ((1 - p) ** (c - 1)) - (1 - p))
                * (1 / np.expm1(ε) + (1 - p))
                / (J * c**2)
            ),
            ((1 / -np.expm1(-ε) - (1 - p)) * (1 / np.expm1(ε) + (1 - p)) / (J * c**2)),
        ],
    ).squeeze()

    return float(res) if res.size == 1 else res


def _optimal_pool_size(
    p: float,
    ε: float,
) -> float:
    """Lowest variance pool size c, considering optimal Se and Sp (reference implementation)"""

    ξ = -1.59362426004004
    ε_crit = -math.log1p(-math.exp(ξ) / (1 - p))

    if ε > ε_crit:
        return ξ / math.log1p(-p)  # Global minimum
    else:
        return 1 + math.log1p(-math.exp(-ε)) / math.log1p(-p)  # Intersection point


def optimal_pool_size(
    p: FloatLike,
    ε: FloatLike,
) -> FloatLike:
    """Lowest variance pool size c, considering optimal Se and Sp (reference implementation)"""

    p = np.atleast_1d(p)[..., None]
    ε = np.atleast_1d(ε)[None, ...]

    ξ = -1.59362426004004
    ε_crit = -np.log1p(-np.exp(ξ) / (1 - p))

    res = np.where(
        ε > ε_crit,
        ξ / np.log1p(-p),  # Global minimum
        1 + np.log1p(-np.exp(-ε)) / np.log1p(-p),  # Intersection point
    ).squeeze()

    return float(res) if res.size == 1 else res


def _round_pool_size(
    p: float,
    c: float,
    J: int,
    ε: float,
    fixed_N: int | None = None,
) -> int:
    """Returns whichever of floor(c) or ceil(c) has better variance, considering optimal Se and Sp (reference implementation)"""

    if fixed_N:
        var_floor = variance(
            p, np.floor(c), fixed_N / np.floor(c), *optimal_accuracy(p, np.floor(c), ε)
        )
        var_ceil = variance(
            p, np.ceil(c), fixed_N / np.ceil(c), *optimal_accuracy(p, np.ceil(c), ε)
        )
    else:
        var_floor = variance(p, np.floor(c), J, *optimal_accuracy(p, np.floor(c), ε))
        var_ceil = variance(p, np.ceil(c), J, *optimal_accuracy(p, np.ceil(c), ε))

    return int(math.floor(c)) if var_floor < var_ceil else int(math.ceil(c))


def round_pool_size(
    p: FloatLike,
    c: FloatLike,
    J: IntLike,
    ε: FloatLike,
    fixed_N: int | None = None,
) -> IntLike:
    """Returns whichever of floor(c) or ceil(c) has better variance, considering optimal Se and Sp (reference implementation)"""

    # Add singleton dimensions to broadcast properly
    p = np.atleast_1d(p)[..., None, None, None]
    c = np.atleast_1d(c)[None, ..., None, None]
    J = np.atleast_1d(J)[None, None, ..., None]
    ε = np.atleast_1d(ε)[None, None, None, ...]

    c_floor = np.floor(c)
    c_ceil = np.ceil(c)

    Se_floor, Sp_floor = optimal_accuracy(p, c_floor, ε)
    Se_ceil, Sp_ceil = optimal_accuracy(p, c_ceil, ε)

    if fixed_N is not None:
        var_floor = variance(p, np.floor(c), fixed_N / c_floor, Se_floor, Sp_floor)
        var_ceil = variance(p, np.ceil(c), fixed_N / c_ceil, Se_ceil, Sp_ceil)
    else:
        var_floor = variance(p, c_floor, J, Se_floor, Sp_floor)
        var_ceil = variance(p, c_ceil, J, Se_ceil, Sp_ceil)

    res = np.where(
        var_floor < var_ceil,
        c_floor.astype(np.int64),
        c_ceil.astype(np.int64),
    )

    return int(res) if res.size == 1 else res
