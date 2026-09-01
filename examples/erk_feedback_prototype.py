"""
Prototype: ERK-like mechanochemical feedback loop on top of FiniteVoronoiSimulator.

Goal (before building any rendering / ML dataset pipeline):
    Check whether a simple forward-Euler feedback loop between
    - c_i   : an ERK-like signal per cell
    - A0_i  : the preferred area per cell
    reproduces the QUALITATIVE behavior reported in the paper:

    1. Sustained oscillation of cell area over time (not flat/decaying).
    2. Area lags behind the ERK-like signal (cross-correlation peak at
       positive lag), roughly analogous to the paper's ~30 min lag.
    3. Removing active noise (Dr=0, f0=0) kills the oscillation.

Model (forward-Euler discretization of a plausible feedback form;
tune tau_A, tau_c, alpha to match desired qualitative behavior):

    c_i(t+dt)  = c_i(t)  + dt * ( (A_i - mean(A)) - c_i ) / tau_c
    A0_i(t+dt) = A0_i(t) + dt * ( (A0 - alpha * c_i) - A0_i ) / tau_A

No image rendering here -- just scalar time series + plots, so the
feedback dynamics can be sanity-checked cheaply before wiring up the
grayscale-Voronoi rendering + CvT training pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyafv as afv


def run(KA=1.0, alpha=0.5, N=100, steps=1500, dt=0.01, seed=0,
        tau_A=8.0, tau_c=1.0,
        mu=1.0, f0=1.0, Dr=0.3,
        relax_steps=200):
    """Run relaxation + feedback loop, return per-step diagnostics for cell 0."""

    rng = np.random.default_rng(seed)

    phys = afv.PhysicalParams(r=1.0, A0=np.pi, P0=4.8, KA=KA, KP=1.0, Lambda=0.2, seed=seed)

    pts = rng.random((N, 2)) * np.sqrt(N) * 2.0
    theta = 2 * np.pi * rng.random(N) - np.pi

    sim = afv.FiniteVoronoiSimulator(pts, phys)

    # Relax to mechanical equilibrium first (uniform A0, no feedback yet)
    for _ in range(relax_steps):
        diag = sim.build()
        pts = pts + mu * diag["forces"] * dt
        sim.update_positions(pts)

    A0_i = np.full(N, phys.A0)
    c_i = np.zeros(N)

    area_hist = np.empty(steps)
    signal_hist = np.empty(steps)
    A0_hist = np.empty(steps)

    for t in range(steps):
        diag = sim.build()
        forces = diag["forces"]
        areas = diag["areas"]

        # active self-propulsion + conservative force -> move points
        active_v = f0 * np.column_stack((np.cos(theta), np.sin(theta)))
        pts = pts + (mu * forces + active_v) * dt
        theta = theta + np.sqrt(2 * Dr * dt) * rng.standard_normal(N)

        # ERK-like feedback (forward Euler)
        c_i = c_i + dt * ((areas - areas.mean()) - c_i) / tau_c
        A0_i = A0_i + dt * ((phys.A0 - alpha * c_i) - A0_i) / tau_A

        sim.update_positions(pts, A0=A0_i)

        area_hist[t] = areas[0]
        signal_hist[t] = c_i[0]
        A0_hist[t] = A0_i[0]

    return area_hist, signal_hist, A0_hist


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
    import sys
    KA, alpha = 1.0, 1.0
    N = 40
    steps = 400
    relax_steps = 80

    print("Running WITH active noise (Dr=0.3, f0=1.0) ...", flush=True)
    area_on, signal_on, A0_on = run(KA=KA, alpha=alpha, f0=1.0, Dr=0.3,
                                     N=N, steps=steps, relax_steps=relax_steps)

    print("Running WITHOUT active noise (Dr=0.0, f0=0.0) ...", flush=True)
    area_off, signal_off, A0_off = run(KA=KA, alpha=alpha, f0=0.0, Dr=0.0,
                                        N=N, steps=steps, relax_steps=relax_steps)

    lags, corr = cross_correlation(area_on, signal_on, max_lag=100)
    peak_lag = lags[np.argmax(corr)]
    print(f"Peak cross-correlation (area vs signal) at lag = {peak_lag} steps "
          f"(positive means area LAGS the signal, as expected)")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)

    axes[0].plot(area_on, label="area (with noise)")
    axes[0].plot(A0_on, label="A0 (preferred area, with noise)", linestyle="--")
    axes[0].set_title(f"Cell 0 area vs preferred area (KA={KA}, alpha={alpha})")
    axes[0].legend()

    axes[1].plot(area_off, label="area (NO noise)", color="C3")
    axes[1].plot(A0_off, label="A0 (NO noise)", linestyle="--", color="C4")
    axes[1].set_title("Same, with active noise disabled -- expect oscillation to die out")
    axes[1].legend()

    axes[2].plot(lags, corr)
    axes[2].axvline(peak_lag, color="k", linestyle=":")
    axes[2].set_title(f"Cross-correlation(area, signal), peak lag = {peak_lag}")
    axes[2].set_xlabel("lag (steps)")

    plt.tight_layout()
    plt.savefig("erk_feedback_prototype.png", dpi=150)
    print("Saved plot to erk_feedback_prototype.png", flush=True)
