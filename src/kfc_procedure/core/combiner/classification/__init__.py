"""
Classification combiners (KFCProcedure).

Includes:
- MajorityVoteCombiner
- StackingClassifierCombiner
- CobraClassifierCombiner
"""

from .majority_vote import MajorityVoteCombiner
from .stacking import StackingClassifierCombiner
from .combined_classifier import CobraClassifierCombiner

__all__ = [
    "MajorityVoteCombiner",
    "StackingClassifierCombiner",
    "CobraClassifierCombiner",
]