import numpy as np
import numpy.typing as npt

from ..utils import IntLike, FloatLike


def _variance(
    p: float,
    c: int,
    J: int,
    Se: float,
    Sp: float,
) -> float:
    """Asymptotic variance (reference implementation)"""

    r = Se + Sp - 1
    return (
        (Se - r * (1 - p) ** c)
        * (1 - Se + r * (1 - p) ** c)
        / (J * c**2 * r**2 * (1 - p) ** (2 * (c - 1)))
    )


def variance(
    p: FloatLike,
    c: IntLike | FloatLike,  # This results in not-quite-correct variances
    J: IntLike | FloatLike,  # """
    Se: FloatLike,
    Sp: FloatLike,
) -> float | npt.NDArray[np.float64]:
    """Asymptotic variance."""

    # Add singleton dimensions to broadcast properly
    p = np.atleast_1d(p)[:, None, None, None, None]
    c = np.atleast_1d(c)[None, :, None, None, None]
    J = np.atleast_1d(J)[None, None, :, None, None]
    Se = np.atleast_1d(Se)[None, None, None, :, None]
    Sp = np.atleast_1d(Sp)[None, None, None, None, :]

    # Compute asymptotic variance according to formula
    r = Se + Sp - 1
    res = (
        (Se - r * (1 - p) ** c)
        * (1 - Se + r * (1 - p) ** c)
        / (J * c**2 * r**2 * (1 - p) ** (2 * (c - 1)))
    ).squeeze()

    # Return float if only a single value is present
    return float(res) if res.size == 1 else res
