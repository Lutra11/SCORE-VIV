"""Fatigue layer: S-N curves, cycle counting, Miner damage, limit states."""

from .assessment import (
    SNCurve,
    damage_from_history,
    extract_cycles,
    life_from_damage_rate,
    spectral_damage,
    system_limit_state,
)

__all__ = [
    "SNCurve", "extract_cycles", "damage_from_history", "spectral_damage",
    "life_from_damage_rate", "system_limit_state",
]
