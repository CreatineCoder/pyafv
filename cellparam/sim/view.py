"""
Draw a saved simulation run so you can see the tissue.

    python cellparam/sim/view.py sim_data/train/Ka0.10_alpha60_seed0.npz
    python cellparam/sim/view.py <file.npz> --frame 500 --color erk
    python cellparam/sim/view.py <file.npz> --gif --frames 200

Saves a PNG (one frame) or a GIF (animation) next to where you run it. Cells are
coloured by area (default), ERK level, or preferred area.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
import numpy as np
from scipy.spatial import Voronoi

import pyafv as afv

COLOR_FIELDS = {"area": ("areas", "cell area", "viridis"),
                "erk": ("erk", "ERK", "coolwarm"),
                "A0": ("A0", "preferred area", "magma")}


def cell_polygons(pts, L, margin=1.5):
    """Periodic Voronoi polygons, including wrapped copies near the box so edge cells draw whole.

    Returns the polygons and, for each, the index of the original cell it belongs to.
    """
    tiled, idx = afv.tile_pbc(pts, L, L / 4)
    vor = Voronoi(tiled)
    near = np.flatnonzero(np.all((tiled > -margin) & (tiled < L + margin), axis=1))
    return [vor.vertices[vor.regions[vor.point_region[i]]] for i in near], idx[near]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--color", choices=list(COLOR_FIELDS), default="area")
    ap.add_argument("--gif", action="store_true", help="save an animation instead of one frame")
    ap.add_argument("--frames", type=int, default=150, help="number of frames in the GIF")
    ap.add_argument("--every", type=int, default=2, help="use every n-th saved frame in the GIF")
    args = ap.parse_args()

    d = np.load(args.file)
    key, label, cmap = COLOR_FIELDS[args.color]
    values, pts_all, L = d[key], d["pts"], float(d["L"])
    lo, hi = np.percentile(values, [1, 99])
    title = f"Ka={float(d['Ka']):g}, alpha={float(d['alpha']):g}, seed={int(d['seed'])}"
    minutes = float(d["frame_minutes"])

    fig, ax = plt.subplots(figsize=(6.4, 6))
    coll = PolyCollection([], cmap=cmap, edgecolors="k", linewidths=0.3)
    coll.set_clim(lo, hi)
    ax.add_collection(coll)
    ax.set(xlim=(0, L), ylim=(0, L), aspect="equal", xticks=[], yticks=[])
    fig.colorbar(coll, ax=ax, label=label, shrink=0.8)

    def draw(f):
        polys, owner = cell_polygons(pts_all[f], L)
        coll.set_verts(polys)
        coll.set_array(values[f][owner])
        ax.set_title(f"{title}\nframe {f}  ({f * minutes / 60:.1f} h)")
        return (coll,)

    name = os.path.splitext(os.path.basename(args.file))[0]
    if args.gif:
        idx = list(range(0, min(len(pts_all), args.frames * args.every), args.every))
        out = f"{name}_{args.color}.gif"
        FuncAnimation(fig, draw, frames=idx, blit=False).save(out, writer=PillowWriter(fps=12), dpi=80)
    else:
        draw(args.frame)
        out = f"{name}_{args.color}_frame{args.frame}.png"
        fig.savefig(out, dpi=120)
    print(f"Saved {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
