"""
Prototype: ERK mechanochemical feedback loop on top of FiniteVoronoiSimulator.

Goal (before building any rendering / ML dataset pipeline):
    Check whether the spec's feedback loop between
    - E_i   : ERK activity per cell
    - A0_i  : the preferred area per cell
    reproduces the QUALITATIVE behavior reported in the paper:

    1. Sustained oscillation of cell area over time (not flat/decaying).
    2. ERK lags behind area (cross-correlation peak at positive lag),
       roughly analogous to the paper's ~30 min lag.
    3. Removing active noise (Dr=0, f0=0) kills the oscillation.

Model (Implementation_Spec.md, Sec. 2.2), integrated with forward Euler in
non-dimensional form, i.e. areas are measured relative to the mean:

    a_i  = A_i / <A>                      (so the mean cell area is 1)
    s_i  = A0_i / A0_ref                  (preferred area, 1 = unperturbed)

    tau_A ds_i/dt = (1 - s_i) - alpha * E_i
    tau_E dE_i/dt = -E_i - E_i**3 + beta * (a_i - 1)

The mechanics stay in PyAFV's natural units (A0_ref = pi, r = 1); only the
feedback variables are non-dimensional, which is what lets alpha and beta
keep their spec meaning independent of the simulator's length scale.

With the spec's alpha ~ 60-100, E must stay ~1e-2 or smaller for s_i to stay
positive, so beta has to be small and the -E**3 term is effectively inactive.

No image rendering here -- just scalar time series + plots, so the
feedback dynamics can be sanity-checked cheaply before wiring up the
grayscale-Voronoi rendering + CvT training pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyafv as afv


def run(KA=0.3, alpha=80.0, beta=0.01, N=60, steps=5000, dt=0.01, seed=0,
        tau_A=1.0, tau_E=1.0,
        mu=1.0, f0=0.3, Dr=0.3,
        relax_steps=1000):
    """Run relaxation + feedback loop, return per-step diagnostics for cell 0.

    Returned area and preferred area are normalised (a_i and s_i above).
    """

    rng = np.random.default_rng(seed)

    phys = afv.PhysicalParams(r=1.0, A0=np.pi, P0=4.8, KA=KA, KP=1.0, Lambda=0.2, seed=seed)
    A0_ref = phys.A0

    pts = rng.random((N, 2)) * np.sqrt(N) * 1.6
    theta = 2 * np.pi * rng.random(N) - np.pi

    sim = afv.FiniteVoronoiSimulator(pts, phys)

    # Relax to mechanical equilibrium first (uniform A0, no feedback yet)
    for _ in range(relax_steps):
        diag = sim.build()
        pts = pts + mu * diag["forces"] * dt
        sim.update_positions(pts)

    s_i = np.ones(N)
    E_i = np.zeros(N)

    area_hist = np.empty(steps)
    erk_hist = np.empty(steps)
    s_hist = np.empty(steps)

    for t in range(steps):
        diag = sim.build()
        forces = diag["forces"]
        a_i = diag["areas"] / diag["areas"].mean()

        # active self-propulsion + conservative force -> move points
        active_v = f0 * np.column_stack((np.cos(theta), np.sin(theta)))
        pts = pts + (mu * forces + active_v) * dt
        theta = theta + np.sqrt(2 * Dr * dt) * rng.standard_normal(N)

        # ERK feedback (forward Euler, both updates use the old E_i)
        dE = (-E_i - E_i**3 + beta * (a_i - 1.0)) / tau_E
        ds = ((1.0 - s_i) - alpha * E_i) / tau_A
        E_i = E_i + dt * dE
        s_i = s_i + dt * ds

        if s_i.min() <= 0:
            raise RuntimeError(f"preferred area went non-positive at step {t} "
                               f"(min s_i = {s_i.min():.3f}); reduce beta or alpha")

        sim.update_positions(pts, A0=A0_ref * s_i)

        area_hist[t] = a_i[0]
        erk_hist[t] = E_i[0]
        s_hist[t] = s_i[0]

    return area_hist, erk_hist, s_hist


def cross_correlation(x, y, max_lag=100):
    """Return lags and normalized cross-correlation of x relative to y (x lagging y)."""
    x = x - x.mean()
    y = y - y.mean()
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([
        np.corrcoef(x[lag:], y[:len(y) - lag])[0, 1] if lag > 0 else
        np.corrcoef(x[:len(x) + lag], y[-lag:])[0, 1] if lag < 0 else
        np.corrcoef(x, y)[0, 1]
        for lag in lags
    ])
    return lags, corr


if __name__ == "__main__":
    KA, alpha = 0.3, 80.0
    dt = 0.01
    common = dict(KA=KA, alpha=alpha, N=60, steps=5000, dt=dt, relax_steps=1000)

    print("Running WITH active noise ...", flush=True)
    area_on, erk_on, s_on = run(**common)

    print("Running WITHOUT active noise (Dr=0.0, f0=0.0) ...", flush=True)
    area_off, erk_off, s_off = run(f0=0.0, Dr=0.0, **common)

    lags, corr = cross_correlation(erk_on, area_on, max_lag=500)
    peak_lag = lags[np.argmax(corr)]
    print(f"Peak cross-correlation (ERK vs area) at lag = {peak_lag} steps = {peak_lag * dt:.2f} time units "
          f"(positive means ERK LAGS area, as expected)")

    time = np.arange(len(area_on)) * dt
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)

    axes[0].plot(time, area_on, label="area / mean area (with noise)")
    axes[0].plot(time, s_on, label="preferred area / A0_ref (with noise)", linestyle="--")
    axes[0].set_title(f"Cell 0 area vs preferred area (KA={KA}, alpha={alpha})")
    axes[0].legend()

    axes[1].plot(time, area_off, label="area / mean area (NO noise)", color="C3")
    axes[1].plot(time, s_off, label="preferred area / A0_ref (NO noise)", linestyle="--", color="C4")
    axes[1].set_title("Same, with active noise disabled -- expect oscillation to die out")
    axes[1].set_xlabel("time")
    axes[1].legend()

    axes[2].plot(lags * dt, corr)
    axes[2].axvline(peak_lag * dt, color="k", linestyle=":")
    axes[2].set_title(f"Cross-correlation(ERK, area), peak lag = {peak_lag * dt:.2f}")
    axes[2].set_xlabel("lag (time units)")

    plt.tight_layout()
    plt.savefig("erk_feedback_prototype.png", dpi=150)
    print("Saved plot to erk_feedback_prototype.png", flush=True)
