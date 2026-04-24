# Author: Bradley R. Kinnard
"""
Beliefs package for conflict resolution and management.
"""

from .conflict_resolution import (
    Belief,
    ResolutionAction,
    ResolutionCase,
    ConflictResolver,
    generate_test_cases,
)

__all__ = [
    "Belief",
    "ResolutionAction",
    "ResolutionCase",
    "ConflictResolver",
    "generate_test_cases",
]
