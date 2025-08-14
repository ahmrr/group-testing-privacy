import numpy as np
import numpy.typing as npt

# Types for vectorized functions that support both NumPy and standard datatypes
IntLike = int | npt.NDArray[np.int64]
FloatLike = float | npt.NDArray[np.float64]
