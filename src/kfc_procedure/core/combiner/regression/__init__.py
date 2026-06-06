"""
Regression combiners (KFCProcedure).

Includes:
- MeanCombiner
- WeightedMeanCombiner
- StackingRegressorCombiner
- GradientCOBRACombiner
- MixCOBRACombiner
"""

from .mean import MeanCombiner
from .weighted_mean import WeightedMeanCombiner
from .stacking import StackingRegressorCombiner
from .gradientcobra import GradientCOBRACombiner
from .mixcobra import MixCOBRACombiner

__all__ = [
    "MeanCombiner",
    "WeightedMeanCombiner",
    "StackingRegressorCombiner",
    "GradientCOBRACombiner",
    "MixCOBRACombiner",
]