"""
EO Brain Analysis Algorithms

Provides spatial anomaly detection (LISA) and flood mapping capabilities
with lazy execution for large-scale satellite imagery processing.
"""

from eo_brain.algorithms.base import (
    # Core abstractions
    Algorithm,
    AlgorithmResult,
    
    # Enums
    Severity,
    MoranQuadrant,
    
    # Configurations
    LISAConfig,
    FloodConfig,
    
    # Algorithm implementations
    LISAAlgorithm,
    NDVIAlgorithm,
    NDWIAlgorithm,
    MNDWIAlgorithm,
    FloodDetectionAlgorithm,
    
    # Registry
    ALGORITHMS,
    get_algorithm,
)

__all__ = [
    # Core
    "Algorithm",
    "AlgorithmResult",
    
    # Enums
    "Severity",
    "MoranQuadrant",
    
    # Config
    "LISAConfig",
    "FloodConfig",
    
    # Algorithms
    "LISAAlgorithm",
    "NDVIAlgorithm",
    "NDWIAlgorithm",
    "MNDWIAlgorithm",
    "FloodDetectionAlgorithm",
    
    # Registry
    "ALGORITHMS",
    "get_algorithm",
]