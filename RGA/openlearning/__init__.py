"""
core包 | core Package
====================
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import warnings

__version__ = "0.0.9"
__author__ = "Open-learning Team"

try:
    from config import RGAConfig
    from config import VRegularizationParams
    from config import CoreMetricsCalculator
    from config import TriValueBalancer

    from metrics import OneWayValve
    from metrics import EnhancedEmbeddingLayer

    from valve import ChainReactionUnit_Final

    from memory import GeologicalMemory
    from memory import SandwichFusion

    from zixin import test_complete_architecture

    from rga import RuleGovernedArchitecture
    from rga import SmartTextDataset
    from rga import VisualTrainingProgress
    from rga import AdvancedConstrainedArchitectureTrainer

except ImportError:
    from .config import RGAConfig
    from .config import VRegularizationParams
    from .config import CoreMetricsCalculator
    from .config import TriValueBalancer

    from .metrics import OneWayValve
    from .metrics import EnhancedEmbeddingLayer

    from .valve import ChainReactionUnit_Final

    from .memory import GeologicalMemory
    from .memory import SandwichFusion

    from .zixin import test_complete_architecture

    from .rga import RuleGovernedArchitecture
    from .rga import SmartTextDataset
    from .rga import VisualTrainingProgress
    from .rga import AdvancedConstrainedArchitectureTrainer



__all__ = [
    "RGAConfig",
    "VRegularizationParams",
    "CoreMetricsCalculator",
    "TriValueBalancer",

    'OneWayValve',
    'EnhancedEmbeddingLayer',

    'ChainReactionUnit_Final',

    'GeologicalMemory',
    'SandwichFusion',

    'test_complete_architecture',

    'RuleGovernedArchitecture',
    'SmartTextDataset',
    'VisualTrainingProgress',
    'AdvancedConstrainedArchitectureTrainer',
]