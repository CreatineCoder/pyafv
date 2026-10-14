"""
Dataset generation (Implementation_Spec.md, Sec. 2.6). One .npz per simulation run.

    python cellparam/sim/generate.py --split train --out sim_data
    python cellparam/sim/generate.py --split train --out sim_data --smoke   # tiny, minutes

Splits:
    train      Ka x alpha training grid, seeds 0-4
    test       Ka at 0.05 steps between training values, unseen alpha, fresh seeds
    extrap     Ka outside the training range (0.05, 0.6)
    stability  one long run at Ka=0.3, alpha=80 (sliced into windows later)

Each file holds per-frame cell centres, areas, ERK and preferred areas plus the
labels and every simulator parameter. Existing files are skipped, so an
interrupted sweep can be resumed by rerunning the same command.

Run as a script, not with -m, so the installed (compiled) pyafv is used.
"""

import argparse
import itertools
import json
import os
import time
import warnings
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

from tissue import ERKTissue, TissueParams

KA_TRAIN = [0.1, 0.2, 0.3, 0.4, 0.5]
ALPHA_TRAIN = [60, 70, 80, 90, 100]
SEEDS = range(5)
FRAMES = 1500
DT_FRAME_MIN = 5.0

# Time-unit calibration from sanity check 4 (ERK should lag area by ~30 min):
# minutes_per_time_unit = 30 / measured lag. Re-measure with checks.py if parameters change.
MINUTES_PER_TIME_UNIT = 33.3

RELAX_STEPS = 500
BURN_IN_TIME = 20.0     # simulation time units of activity discarded before recording


def split_jobs(split):
    if split == "train":
        grid = itertools.product(KA_TRAIN, ALPHA_TRAIN, SEEDS)
    elif split == "test":
        grid = itertools.product([0.15, 0.25, 0.35, 0.45], [65, 75, 85, 95], range(100, 102))
    elif split == "extrap":
        grid = itertools.product([0.05, 0.6], [80], range(200, 202))
    elif split == "stability":
        grid = [(0.3, 80, 300)]
    else:
        raise ValueError(split)
    return list(grid)


def run_one(job):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    Ka, alpha, seed, base, frames, out_dir = job
    path = os.path.join(out_dir, f"Ka{Ka:.2f}_alpha{alpha:g}_seed{seed}.npz")
    if os.path.exists(path):
        return path, "skipped"

    params = replace(base, Ka=Ka, alpha=float(alpha))
    spf = max(1, round(DT_FRAME_MIN / MINUTES_PER_TIME_UNIT / params.dt))
    t0 = time.time()
    try:
        T = ERKTissue(params, seed=seed)
        T.relax(RELAX_STEPS)
        T.record(max(1, round(BURN_IN_TIME / (spf * params.dt))), spf)
        rec = T.record(frames, spf)
    except Exception as e:
        return path, f"FAILED after {time.time() - t0:.0f}s: {type(e).__name__}: {e}"
    clamp_fraction = T.clamped_cell_steps / max(T.feedback_cell_steps, 1)

    tmp = path + ".part.npz"
    np.savez_compressed(
        tmp,
        pts=rec["pts"].astype(np.float32),
        areas=rec["areas"].astype(np.float32),
        erk=rec["erk"].astype(np.float32),
        A0=rec["A0"].astype(np.float32),
        time=rec["time"],
        Ka=Ka, alpha=alpha, seed=seed, L=T.L,
        frame_minutes=spf * params.dt * MINUTES_PER_TIME_UNIT,
        A0_clamp_fraction=clamp_fraction,
        params=json.dumps(params.to_dict()),
    )
    os.replace(tmp, path)
    note = f"  (A0 floor used in {clamp_fraction:.1e} of cell-steps)" if clamp_fraction else ""
    return path, f"{time.time() - t0:.0f}s{note}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "extrap", "stability"], default="train")
    ap.add_argument("--out", default="sim_data")
    ap.add_argument("--N", type=int, default=300)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true", help="2 runs, 40 cells, 20 frames: checks the pipeline only")
    args = ap.parse_args()

    base = TissueParams(N=args.N)
    frames = args.frames or (238 if args.split == "stability" else FRAMES)
    jobs = split_jobs(args.split)
    if args.smoke:
        base, frames, jobs = replace(base, N=40), 20, jobs[:2]

    out_dir = os.path.join(args.out, args.split)
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(jobs)} runs -> {out_dir} (N={base.N}, {frames} frames, {args.workers} workers)", flush=True)
    failed = []
    with Pool(args.workers) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(
                run_one, [(*j, base, frames, out_dir) for j in jobs]), 1):
            print(f"[{i}/{len(jobs)}] {os.path.basename(path)}  {status}", flush=True)
            if status.startswith("FAILED"):
                failed.append(os.path.basename(path))
    if failed:
        print(f"\n{len(failed)} run(s) failed and were not saved: {', '.join(failed)}\n"
              "Rerun the same command to retry them (finished files are skipped).", flush=True)
    else:
        print("\nAll runs finished.", flush=True)


if __name__ == "__main__":
    main()
