"""
Periodic AFV tissue with ERK mechanochemical feedback (Implementation_Spec.md, Sec. 2.2).

Units follow the spec: lengths are scaled so the mean cell area is 1, so A_i
hovers around 1 and the box side is L = sqrt(N).

Energy (PyAFV form, Bi et al. SPV-style perimeter term):
    U = sum_i  KA_afv (A_i - A0_i)^2 + KP (P_i - p0)^2 + Lambda * (free arc length)_i
The spec writes the area term as (1/2) Ka (A - A0)^2, so KA_afv = Ka / 2.

Feedback, per cell, with s_i = A0_i (the preferred area, 1 = unperturbed):
    tau_A ds_i/dt = (1 - s_i) - (alpha / alpha_scale) * E_i
    tau_E dE_i/dt = -E_i - E_i^3 + beta * (A_i / <A> - 1)
(see TissueParams.alpha_scale for why alpha is rescaled), with s_i floored at A0_min.

Periodic boundaries use pyafv.tile_pbc: cells within 4r of an edge are copied
to the opposite side each step, which gives correct shapes AND forces for the
original N cells (the copies are discarded after build()).
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.spatial import Voronoi
import pyafv as afv


@dataclass
class TissueParams:
    N: int = 300
    Ka: float = 0.3          # spec area stiffness (energy (1/2) Ka (A - A0)^2)
    KP: float = 0.2
    p0: float = 3.8          # preferred perimeter (= shape index, since mean area is 1)
    Lambda: float = 0.2
    r: float = 0.75          # max cell radius; caps area at pi r^2 ~ 1.77
    mu: float = 1.0          # mobility (1 / eta)
    f0: float = 0.1          # self-propulsion speed (zeta / eta)
    Dr: float = 0.3          # rotational diffusion: dphi ~ N(0, 2 Dr dt)
    feedback: bool = True
    alpha: float = 80.0
    # ASSUMPTION (verify against the paper): alpha = 60-100 is quoted in physical area units
    # (MDCK cells are ~100-300 um^2). Taken literally with mean area 1, alpha*E drives A0 negative
    # unless beta is so small that the -E^3 term never matters, so we use alpha / alpha_scale.
    alpha_scale: float = 200.0
    beta: float = 60.0
    tau_A: float = 3.0
    tau_E: float = 3.0
    # Floor on the preferred area: a cell cannot "want" a non-positive size. Only reached in
    # rare excursions of the softest, most strongly coupled runs (Ka=0.1, alpha>=80); how
    # often it binds is counted in ERKTissue.clamped_cell_steps.
    A0_min: float = 0.1
    dt: float = 0.01

    def afv_params(self, seed=None) -> afv.PhysicalParams:
        return afv.PhysicalParams(r=self.r, A0=1.0, P0=self.p0, KA=self.Ka / 2, KP=self.KP,
                                  Lambda=self.Lambda, seed=seed)

    def to_dict(self):
        return asdict(self)


def lloyd_points(N, L, rng, iters=20):
    """Random points in a periodic LxL box, made more uniform by Lloyd iterations."""
    pts = rng.random((N, 2)) * L
    for _ in range(iters):
        tiled, _ = afv.tile_pbc(pts, L, L / 4)
        vor = Voronoi(tiled)
        new = np.empty_like(pts)
        for i in range(N):
            poly = vor.vertices[vor.regions[vor.point_region[i]]]
            x, y = poly[:, 0], poly[:, 1]
            cross = x * np.roll(y, -1) - np.roll(x, -1) * y
            area = cross.sum() / 2
            new[i, 0] = ((x + np.roll(x, -1)) * cross).sum() / (6 * area)
            new[i, 1] = ((y + np.roll(y, -1)) * cross).sum() / (6 * area)
        pts = new % L
    return pts


class ERKTissue:
    def __init__(self, params: TissueParams, seed: int = 0):
        self.p = params
        self.N = params.N
        self.L = np.sqrt(params.N)
        self.rng = np.random.default_rng(seed)
        self.phys = params.afv_params(seed)

        self.pts = lloyd_points(self.N, self.L, self.rng)
        self.theta = self.rng.uniform(-np.pi, np.pi, self.N)
        self.s = np.ones(self.N)
        self.E = np.zeros(self.N)
        self.t = 0.0
        self.feedback_cell_steps = 0
        self.clamped_cell_steps = 0
        self.sim = afv.FiniteVoronoiSimulator(self.pts, self.phys)

    def geometry(self):
        """Forces, areas, perimeters and free arc lengths of the N original cells."""
        tiled, idx = afv.tile_pbc(self.pts, self.L, self.p.r)
        self.sim.update_positions(tiled, A0=self.s[idx])
        d = self.sim.build(connect=False)
        N = self.N
        return d["forces"][:N], d["areas"][:N], d["perimeters"][:N], d["arclens"][:N]

    def energy(self, areas, perims, arcs):
        ph = self.phys
        return float(np.sum(ph.KA * (areas - self.s) ** 2 + ph.KP * (perims - ph.P0) ** 2
                            + ph.Lambda * arcs))

    def step(self, active=True, feedback=None):
        """Advance one dt; returns this step's (areas, energy) measured before the move."""
        p = self.p
        feedback = p.feedback if feedback is None else feedback
        F, A, P, arc = self.geometry()

        v = p.mu * F
        if active:
            v = v + p.f0 * np.column_stack((np.cos(self.theta), np.sin(self.theta)))
            self.theta += np.sqrt(2 * p.Dr * p.dt) * self.rng.standard_normal(self.N)
        self.pts = (self.pts + v * p.dt) % self.L

        if feedback:
            a = A / A.mean()
            dE = (-self.E - self.E ** 3 + p.beta * (a - 1.0)) / p.tau_E
            ds = ((1.0 - self.s) - p.alpha / p.alpha_scale * self.E) / p.tau_A
            self.E += p.dt * dE
            self.s += p.dt * ds
            low = self.s < p.A0_min
            self.s[low] = p.A0_min
            self.clamped_cell_steps += int(low.sum())
            self.feedback_cell_steps += self.N

        self.t += p.dt
        return A, self.energy(A, P, arc)

    def relax(self, steps):
        """Passive relaxation (no activity, no feedback); returns the energy trace."""
        return np.array([self.step(active=False, feedback=False)[1] for _ in range(steps)])

    def record(self, n_frames, steps_per_frame, active=True):
        """Run and save one frame every steps_per_frame steps."""
        out = {k: np.empty((n_frames, self.N)) for k in ("areas", "erk", "A0")}
        out["pts"] = np.empty((n_frames, self.N, 2))
        out["time"] = np.empty(n_frames)
        for f in range(n_frames):
            for _ in range(steps_per_frame):
                A, _ = self.step(active=active)
            out["areas"][f] = A
            out["erk"][f] = self.E
            out["A0"][f] = self.s
            out["pts"][f] = self.pts
            out["time"][f] = self.t
        return out
