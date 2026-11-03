"""Euler-Bernoulli beam FEM with Hermite elements.

Hermite cubic beam element (w, theta per node), consistent mass matrix,
geometric stiffness from axial force, translational springs, static and
modal solvers, curvature/stress recovery, and mesh-convergence helpers.

Closed-form references used by the validation tests:
- simply supported:   f_n = n^2 pi / (2 L^2) * sqrt(EI/m)
- cantilever:         beta_1 L = 1.875
- clamped-clamped:    beta_1 L = 4.730
- cantilever tip deflection: P L^3 / (3 EI), uniform q: q L^4 / (8 EI)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh as general_eigh


@dataclass
class Support:
    """Support at axial coordinate ``x``.

    kind: "pin" (w=0), "clamp" (w=theta=0), or "spring" (w-DOF spring).
    """

    x: float
    kind: str = "pin"
    stiffness: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("pin", "clamp", "spring"):
            raise ValueError(f"support kind must be pin|clamp|spring: {self.kind}")
        if self.kind == "spring" and self.stiffness <= 0:
            raise ValueError("spring support needs positive stiffness")


@dataclass
class BeamModel:
    """Beam definition (uniform EI and mass per length)."""

    length: float
    EI: float
    mass_per_length: float
    axial_force: float = 0.0  # tension positive
    supports: list = field(default_factory=list)
    outer_diameter: float | None = None
    elastic_modulus: float | None = None  # for stress recovery; None -> moment only

    def validate(self) -> None:
        for name, val in (("length", self.length), ("EI", self.EI),
                          ("mass_per_length", self.mass_per_length)):
            if val <= 0:
                raise ValueError(f"{name} must be positive: {val}")
        for s in self.supports:
            if not (0.0 <= s.x <= self.length):
                raise ValueError(f"support outside beam: {s.x}")
        if self.elastic_modulus is not None and self.elastic_modulus <= 0:
            raise ValueError("elastic_modulus must be positive")


# ---- element matrices ----------------------------------------------------


def hermite_k(EI: float, L: float) -> np.ndarray:
    return (EI / L**3) * np.array([
        [12.0, 6 * L, -12.0, 6 * L],
        [6 * L, 4 * L**2, -6 * L, 2 * L**2],
        [-12.0, -6 * L, 12.0, -6 * L],
        [6 * L, 2 * L**2, -6 * L, 4 * L**2],
    ])


def hermite_m(mass: float, L: float) -> np.ndarray:
    return (mass * L / 420.0) * np.array([
        [156.0, 22 * L, 54.0, -13 * L],
        [22 * L, 4 * L**2, 13 * L, -3 * L**2],
        [54.0, 13 * L, 156.0, -22 * L],
        [-13 * L, -3 * L**2, -22 * L, 4 * L**2],
    ])


def hermite_kg(T: float, L: float) -> np.ndarray:
    """Geometric stiffness (tension T > 0 stiffens the beam)."""
    return (T / (30.0 * L)) * np.array([
        [36.0, 3 * L, -36.0, 3 * L],
        [3 * L, 4 * L**2, -3 * L, -L**2],
        [-36.0, -3 * L, 36.0, -3 * L],
        [3 * L, -L**2, -3 * L, 4 * L**2],
    ])


def consistent_q(q: float, L: float) -> np.ndarray:
    """Consistent nodal loads for uniform distributed load q."""
    return np.array([q * L / 2, q * L**2 / 12, q * L / 2, -q * L**2 / 12])


def curvature_B(xi: float, L: float) -> np.ndarray:
    """Second derivatives of Hermite shape functions at local coord xi in [0, L]."""
    return np.array([
        -6.0 / L**2 + 12 * xi / L**3,
        -4.0 / L + 6 * xi**2 / L**3,
        6.0 / L**2 - 12 * xi / L**3,
        -2.0 / L + 6 * xi**2 / L**3,
    ])


# ---- assembled model ------------------------------------------------------


class BeamFEM:
    """Assembled beam FEM with support elimination."""

    def __init__(self, model: BeamModel, n_elements: int):
        model.validate()
        if n_elements < 1:
            raise ValueError("n_elements must be >= 1")
        self.model = model
        self.n_elements = int(n_elements)
        self.n_nodes = self.n_elements + 1
        self.n_dof = 2 * self.n_nodes
        self.x = np.linspace(0.0, model.length, self.n_nodes)
        self._build()

    # -- assembly -----------------------------------------------------------

    def _build(self) -> None:
        m = self.model
        K = np.zeros((self.n_dof, self.n_dof))
        M = np.zeros((self.n_dof, self.n_dof))
        Kg = np.zeros((self.n_dof, self.n_dof))
        self._elem_nodes = []
        for e in range(self.n_elements):
            ia, ib = e, e + 1
            L = self.x[ib] - self.x[ia]
            dofs = np.array([2 * ia, 2 * ia + 1, 2 * ib, 2 * ib + 1])
            K[np.ix_(dofs, dofs)] += hermite_k(m.EI, L)
            M[np.ix_(dofs, dofs)] += hermite_m(m.mass_per_length, L)
            Kg[np.ix_(dofs, dofs)] += hermite_kg(m.axial_force, L)
            self._elem_nodes.append((dofs, L, 0.5 * (self.x[ia] + self.x[ib])))
        # springs add translational stiffness at nearest node
        self.support_snap = []
        for s in m.supports:
            node = int(np.argmin(np.abs(self.x - s.x)))
            snap_err = abs(self.x[node] - s.x)
            dof = 2 * node
            if s.kind == "spring":
                K[dof, dof] += s.stiffness
            self.support_snap.append({
                "requested_x": s.x, "node_x": float(self.x[node]),
                "node": node, "kind": s.kind, "snap_error_m": float(snap_err),
                "label": s.label or s.kind,
            })
        self.K_full, self.M_full, self.Kg_full = K, M, Kg
        self._reduce()

    def _reduce(self) -> None:
        fixed = set()
        for s in self.support_snap:
            if s["kind"] in ("pin",):
                fixed.add(2 * s["node"])
            elif s["kind"] == "clamp":
                fixed.add(2 * s["node"])
                fixed.add(2 * s["node"] + 1)
        free = [d for d in range(self.n_dof) if d not in fixed]
        self.free = np.array(free, dtype=int)
        self.K = self.K_full[np.ix_(self.free, self.free)]
        self.M = self.M_full[np.ix_(self.free, self.free)]
        self.Kg = self.Kg_full[np.ix_(self.free, self.free)]

    # -- solvers -------------------------------------------------------------

    def natural_modes(self, n_modes: int = 3):
        """Return (freqs_hz, modes) with modes on all nodes (constrained = 0)."""
        k = min(n_modes, len(self.free))
        vals, vecs = general_eigh(self.K + self.Kg, self.M,
                                 subset_by_index=[0, k - 1])
        freqs = np.sqrt(np.maximum(vals, 0.0)) / (2.0 * np.pi)
        shapes = np.zeros((k, self.n_nodes))
        for j in range(k):
            full = np.zeros(self.n_dof)
            full[self.free] = vecs[:, j]
            shapes[j] = full[0::2]
            # normalize: max |w| = 1
            peak = np.max(np.abs(shapes[j]))
            if peak > 0:
                shapes[j] /= peak
        return freqs, shapes

    def static_displacement(self, distributed_q: float = 0.0,
                            point_loads: dict | None = None) -> np.ndarray:
        """Solve K u = f; returns full nodal w (constrained nodes = 0)."""
        f = np.zeros(self.n_dof)
        for e in range(self.n_elements):
            dofs, L, _ = self._elem_nodes[e]
            f[dofs] += consistent_q(distributed_q, L)
        for x_str, load in (point_loads or {}).items():
            node = int(np.argmin(np.abs(self.x - float(x_str))))
            f[2 * node] += float(load)
        fr = f[self.free]
        ur = np.linalg.solve(self.K + self.Kg, fr)
        full = np.zeros(self.n_dof)
        full[self.free] = ur
        return full[0::2]

    # -- recovery -------------------------------------------------------------

    def mode_curvature(self, shape: np.ndarray) -> np.ndarray:
        """Curvature of a nodal mode shape via cubic spline (max-normalized)."""
        if shape.shape != self.x.shape:
            raise ValueError("shape must match node grid")
        cs = CubicSpline(self.x, shape)
        return cs(self.x, 2)

    def bending_stress(self, shape: np.ndarray, scale: float) -> np.ndarray | None:
        """Stress = E * c * kappa * scale; None when E or D is unknown."""
        m = self.model
        if m.elastic_modulus is None or m.outer_diameter is None:
            return None
        kappa = self.mode_curvature(shape)
        c = 0.5 * m.outer_diameter
        return m.elastic_modulus * c * kappa * scale


# ---- mesh convergence ------------------------------------------------------


def mesh_convergence(builder, n_elements_list, n_modes: int = 1):
    """builder(n_elements) -> BeamFEM; returns convergence rows."""
    rows = []
    ref = None
    for n in n_elements_list:
        fem = builder(int(n))
        freqs, _ = fem.natural_modes(n_modes=n_modes)
        row = {"n_elements": int(n)}
        for j in range(n_modes):
            row[f"f{j + 1}_hz"] = float(freqs[j])
        if ref is None and n == n_elements_list[-1]:
            ref = freqs
        rows.append(row)
    ref = rows[-1]
    for row in rows:
        for j in range(n_modes):
            key = f"f{j + 1}_hz"
            row[f"rel_err_{j + 1}"] = abs(row[key] - ref[key]) / abs(ref[key])
    return rows


# ---- analytic references ----------------------------------------------------


def ss_beam_frequencies(EI, mass, L, n=3) -> np.ndarray:
    k = np.arange(1, n + 1)
    return (k**2) * np.pi / (2 * L**2) * np.sqrt(EI / mass)


def cantilever_frequencies(EI, mass, L, n=3) -> np.ndarray:
    beta = np.array([1.875104, 4.694091, 7.854757])[:n]
    return (beta / L) ** 2 * np.sqrt(EI / mass) / (2 * np.pi)


def clamped_clamped_frequencies(EI, mass, L, n=3) -> np.ndarray:
    beta = np.array([4.730041, 7.853205, 10.995607])[:n]
    return (beta / L) ** 2 * np.sqrt(EI / mass) / (2 * np.pi)
