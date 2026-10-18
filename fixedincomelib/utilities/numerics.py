import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        axis1 = np.array(axis1, dtype=float, copy=True)
        values = np.array(values, dtype=float, copy=True)
        if axis1.ndim != 1 or values.ndim != 1:
            raise ValueError('Knots and ordinates must be one-dimensional.')
        if len(axis1) == 0 or len(axis1) != len(values):
            raise ValueError('Use nonempty knots and equally many ordinates.')
        if not np.all(np.isfinite(axis1)) or not np.all(np.isfinite(values)):
            raise ValueError('Knots and ordinates must be finite.')
        if not np.all(np.diff(axis1) > 0):
            raise ValueError('Knots must be strictly increasing.')
        if extrapolation_method != ExtrapMethod.FLAT:
            raise ValueError('This interpolator supports only flat extrapolation.')
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)

    def interpolate(self, x: float) -> float:
        if not np.isfinite(x):
            raise ValueError('The interpolation point must be finite.')
        # The first knot >= x selects the value on (x[i-1], x[i]].
        index = min(np.searchsorted(self.axis1_, x, side='left'), self.length_ - 1)
        return float(self.values_[index])

    def integrate(self, start_x: float, end_x: float) -> float:
        weights = self.gradient_of_integrated_value_wrt_ordinate(start_x, end_x)
        return float(np.dot(weights, self.values_))

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        if not np.isfinite(x):
            raise ValueError('The interpolation point must be finite.')
        index = min(np.searchsorted(self.axis1_, x, side='left'), self.length_ - 1)
        gradient = np.zeros(self.length_, dtype=float)
        gradient[index] = 1.0
        return gradient

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        if not np.isfinite(start_x) or not np.isfinite(end_x):
            raise ValueError('Integration endpoints must be finite.')
        lower, upper = min(start_x, end_x), max(start_x, end_x)
        sign = 1.0 if end_x >= start_x else -1.0
        # Each entry is the length of the interval using that ordinate.
        # The outer intervals include the flat extrapolation wings.
        left_edges = np.concatenate(([-np.inf], self.axis1_[:-1]))
        right_edges = np.concatenate((self.axis1_[:-1], [np.inf]))
        lengths = np.maximum(
            0.0, np.minimum(upper, right_edges) - np.maximum(lower, left_edges)
        )
        return sign * lengths



class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')
