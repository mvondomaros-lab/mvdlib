from collections.abc import Callable
from typing import TypeAlias

ScalarFloatFunc: TypeAlias = Callable[[float], float]
NumbaScalarFloatFunc: TypeAlias = Callable[[float], float]
