"""
Stage 1 sanity checks (Implementation_Spec.md, Sec. 2.5). Run from anywhere:

    python cellparam/sim/checks.py              # N=100 quick version
    python cellparam/sim/checks.py --N 300      # spec size (slower)

Prints PASS/FAIL per check and saves sanity_checks.png (and a JSON summary)
into --out. Run as a script, not with -m, so the installed (compiled) pyafv is
imported instead of the slower uncompiled source in this repo.
"""

import argparse
import json
import os
import warnings
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

from tissue import ERKTissue, TissueParams
from analysis import amplitude, cross_correlation, oscillation_score

KA_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5]
TARGET_LAG_MIN = 30.0


def _simulate(job):
    """Relax, burn in, record with noise; optionally continue with noise switched off."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    params, seed, cfg = job
    T = ERKTissue(params, seed=seed)
    energy = T.relax(cfg["relax_steps"])
    T.record(cfg["burn_frames"], cfg["spf"])
    rec = T.record(cfg["frames"], cfg["spf"])
    out = dict(energy=energy, areas=rec["areas"], erk=rec["erk"], time=rec["time"])
    if cfg.get("then_quiet"):
        quiet = T.record(cfg["frames"], cfg["spf"], active=False)
        out["quiet_areas"] = quiet["areas"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--frames", type=int, default=600, help="recorded frames per run")
    ap.add_argument("--spf", type=int, default=10, help="simulation steps per frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--out", default=".")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override TissueParams fields, e.g. --set beta=60 tau_A=3")
    args = ap.parse_args()

    overrides = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}
    base = replace(TissueParams(N=args.N), **overrides)
    cfg = dict(relax_steps=500, burn_frames=100, frames=args.frames, spf=args.spf)
    frame_dt = args.spf * base.dt
    max_lag = int(10 / frame_dt)

    jobs = [(replace(base, Ka=ka), args.seed, dict(cfg, then_quiet=(ka == base.Ka))) for ka in KA_SWEEP]
    jobs.append((replace(base, feedback=False), args.seed, cfg))
    with Pool(args.workers) as pool:
        results = pool.map(_simulate, jobs)
    by_ka = dict(zip(KA_SWEEP, results[:-1]))
    main_run, no_fb = by_ka[base.Ka], results[-1]

    report = {}

    # 1. Pure relaxation must lower the energy monotonically.
    e = main_run["energy"]
    rises = int(np.sum(np.diff(e) > 1e-9 * abs(e[0])))
    report["1_energy_decreases"] = dict(passed=rises == 0, start=float(e[0]), end=float(e[-1]), increases=rises)

    # 2. Feedback + activity give sustained area cycling: the autocorrelation swings negative,
    #    more than in the identical run without feedback, and does not fade over the run.
    A = main_run["areas"]
    half = len(A) // 2
    dip, dip_lag = oscillation_score(A, max_lag)
    dip_nofb, _ = oscillation_score(no_fb["areas"], max_lag)
    sustained = amplitude(A[half:]) > 0.7 * amplitude(A[:half])
    report["2_oscillations"] = dict(passed=bool(dip < -0.2 and dip < dip_nofb and sustained),
                                    autocorr_dip=dip, autocorr_dip_without_feedback=dip_nofb,
                                    period=2 * dip_lag * frame_dt, sustained=bool(sustained))

    # 3. Amplitude must fall monotonically as Ka rises.
    amps = [amplitude(by_ka[ka]["areas"]) for ka in KA_SWEEP]
    report["3_amplitude_vs_Ka"] = dict(passed=bool(np.all(np.diff(amps) < 0)),
                                       Ka=KA_SWEEP, amplitude=amps)

    # 4. ERK lags area. The lag also fixes the time unit: lag should correspond to ~30 min.
    lags, cc = cross_correlation(main_run["erk"], A, max_lag)
    lag_t = float(lags[np.argmax(cc)] * frame_dt)
    report["4_erk_lags_area"] = dict(passed=lag_t > 0, lag_time_units=lag_t,
                                     minutes_per_time_unit=TARGET_LAG_MIN / lag_t if lag_t > 0 else None)

    # 5. Switching activity off (feedback still on) lets the oscillation die out.
    q = main_run["quiet_areas"]
    w = max(len(q) // 5, 1)
    amp_start, amp_end = amplitude(q[:w]), amplitude(q[-w:])
    report["5_noise_free_decays"] = dict(passed=bool(amp_end < 0.2 * amp_start),
                                         amp_after_switch_off=amp_start, amp_at_end=amp_end)

    for name, r in report.items():
        details = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "passed"}
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {name}  {details}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sanity_checks.json"), "w") as f:
        json.dump(dict(params=base.to_dict(), frame_dt=frame_dt, report=report), f, indent=2, default=float)
    _plot(args.out, base, main_run, no_fb, amps, lags * frame_dt, cc, frame_dt, report)


def _plot(out, base, main_run, no_fb, amps, lag_t, cc, frame_dt, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def title(key, text):
        return f"[{'PASS' if report[key]['passed'] else 'FAIL'}] {text}"

    fig, ax = plt.subplots(3, 2, figsize=(12, 11))
    ax = ax.ravel()

    ax[0].plot(main_run["energy"])
    ax[0].set(title=title("1_energy_decreases", "1. Energy during passive relaxation"),
              xlabel="step", ylabel="energy")

    t = main_run["time"] - main_run["time"][0]
    for c in range(3):
        ax[1].plot(t, main_run["areas"][:, c], lw=1, label=f"cell {c}")
    ax[1].plot(t, no_fb["areas"][:, 0], color="grey", lw=1, ls=":", label="cell 0, no feedback")
    ax[1].set(title=title("2_oscillations", f"2. Cell areas, Ka={base.Ka}"), xlabel="time", ylabel="area")
    ax[1].legend(fontsize=8)

    ax[2].plot(KA_SWEEP, amps, "o-")
    ax[2].set(title=title("3_amplitude_vs_Ka", "3. Area fluctuation size vs Ka"), xlabel="Ka",
              ylabel="mean per-cell std of area")

    ax[3].plot(lag_t, cc)
    ax[3].axvline(report["4_erk_lags_area"]["lag_time_units"], color="k", ls=":")
    ax[3].set(title=title("4_erk_lags_area", "4. Corr(ERK(t+lag), area(t))"), xlabel="lag (time units)")

    q = main_run["quiet_areas"]
    tq = np.arange(len(q)) * frame_dt
    ax[4].plot(tq, q[:, 0], lw=1, label="cell 0")
    ax[4].plot(tq, q.std(axis=1), color="k", label="spread across cells")
    ax[4].set(title=title("5_noise_free_decays", "5. After switching activity off"), xlabel="time")
    ax[4].legend(fontsize=8)

    ax[5].axis("off")
    ax[5].text(0, 1, "\n".join(f"{k} = {v}" for k, v in base.to_dict().items()), va="top", family="monospace",
               fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out, "sanity_checks.png"), dpi=120)
    print(f"Saved {os.path.join(out, 'sanity_checks.png')}")


if __name__ == "__main__":
    main()
