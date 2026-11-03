"""Mitigation layer: encoding, constraints, layout repair."""

from .layout import (
    FAMILIES,
    ConstraintSpec,
    MitigationSpec,
    check_constraints,
)

__all__ = ["FAMILIES", "MitigationSpec", "ConstraintSpec", "check_constraints"]
