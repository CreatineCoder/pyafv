# PyAFV

<a href="https://pyafv.github.io"><img src="https://raw.githubusercontent.com/pyafv/assets/main/gif/test.gif" alt="pyafv_pbc" align="right" /></a>

[![PyPi](https://img.shields.io/pypi/v/pyafv?color=brightgreen&cacheSeconds=300)](https://pypi.org/project/pyafv/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyafv.svg)](https://anaconda.org/conda-forge/pyafv)
[![Docker](https://img.shields.io/docker/v/wwang721/pyafv?logo=docker&sort=semver&color=blue&label=docker)](https://hub.docker.com/r/wwang721/pyafv)
[![License](https://img.shields.io/github/license/wwang721/pyafv)](/LICENSE)
[![Tests on all platforms](https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml/badge.svg)](https://github.com/wwang721/pyafv/actions/workflows/tests_all_platform.yml)
[![pytest](https://github.com/wwang721/pyafv/actions/workflows/tests.yml/badge.svg)](https://github.com/wwang721/pyafv/actions/workflows/tests.yml)
[![Codecov](https://codecov.io/github/wwang721/pyafv/branch/main/graph/badge.svg?token=VSXSOX8HVS)](https://codecov.io/github/wwang721/pyafv/tree/main)
[![Documentation](https://app.readthedocs.org/projects/pyafv/badge/?version=latest)](https://pyafv.readthedocs.io)
[![arXiv:2604.15481](https://img.shields.io/badge/arXiv-2604.15481-grey.svg?colorB=a42c25&logo=arxiv)](https://doi.org/10.48550/arXiv.2604.15481)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/index.ipynb)

<!--
[![Downloads](https://img.shields.io/pypi/dm/pyafv.svg?cacheSeconds=43200)](https://pypi.org/project/pyafv/)
[![PhysRevE.109.054408](https://img.shields.io/badge/Phys.%20Rev.%20E-109.054408-grey.svg?colorA=8c6040)](https://doi.org/10.1103/PhysRevE.109.054408)
[![Soft Matter](https://img.shields.io/badge/Soft%20Matter-XXXXX-63a7c2.svg?colorA=63a7c2&colorB=grey)](https://doi.org/10.1103/PhysRevE.109.054408)
-->

**PyAFV** is a Python package for simulating cellular tissues based on the 2D **active finite Voronoi (AFV) model**.
It provides a computational framework for investigating collective cell behaviors such as motility, adhesion, jamming, and tissue fracture in active matter and biophysical systems.
In contrast to standard vertex or Voronoi models, the AFV model incorporates finite interaction ranges and cell-medium interfaces, allowing for detachment, free boundaries, and fragmentation.
The package includes tools for geometry handling, time evolution, and analysis of cell configurations.
The AFV formalism was introduced and further developed in, for example, Refs. [[1](#huang2023bridging)&ndash;[3](#wang2026divergence)].

> See an interactive [simulation demo](https://dapengbi.com/paper_simulation_demos/afv_model/index.html) using **PyAFV** on Prof. Dapeng (Max) Bi's homepage!


## ERK feedback dataset (this fork)

This fork adds a periodic AFV tissue with ERK mechanochemical feedback, used to generate simulated training data for inferring the area stiffness $K_a$ and feedback strength $\alpha$ from cell dynamics.
Everything lives in [`cellparam/sim/`](cellparam/sim):

| File | What it does |
|---|---|
| [`tissue.py`](cellparam/sim/tissue.py) | `ERKTissue` simulator and `TissueParams` (all model parameters and defaults) |
| [`checks.py`](cellparam/sim/checks.py) | Stage 1 sanity checks: energy relaxation, oscillations, amplitude vs $K_a$, ERK–area lag, decay without noise |
| [`generate.py`](cellparam/sim/generate.py) | Generates the dataset, one `.npz` per simulation run |
| [`view.py`](cellparam/sim/view.py) | Draws a saved run as a PNG frame or a GIF |
| [`analysis.py`](cellparam/sim/analysis.py) | Time-series statistics used by the checks |

[`examples/erk_feedback_prototype.py`](examples/erk_feedback_prototype.py) is the earlier single-script prototype of the feedback loop. It is kept for reference only.

### Setup

You need Python 3.10–3.14 (we used 3.12). Clone the fork and install the compiled package from PyPI. Version 0.4.16 is tested and includes `tile_pbc`, which the simulator uses for periodic boundaries.
```bash
git clone https://github.com/CreatineCoder/pyafv.git
cd pyafv
pip install "pyafv>=0.4.16"
```
Run all commands below **from the repository root**, and run them as scripts (`python cellparam/sim/<script>.py`), **not** with `python -m`.
Running them as scripts means Python imports the fast compiled `pyafv` you installed, not the slower pure-Python source in `./pyafv`.

### 1. Sanity checks (optional, a few minutes)

```bash
python cellparam/sim/checks.py                    # N=100 cells, quick
python cellparam/sim/checks.py --N 300            # dataset size, slower
python cellparam/sim/checks.py --set beta=60 tau_A=3   # override any TissueParams field
```
Each check prints `PASS` or `FAIL`. The script writes `sanity_checks.png` and `sanity_checks.json` to `--out` (default: the current directory).
Check 4 also measures how many minutes one simulation time unit represents. If you change model parameters, use that result to update `MINUTES_PER_TIME_UNIT` in `generate.py`.

> With the current defaults, checks 1, 3, 4 and 5 pass. Check 2 (sustained oscillations) fails at N=100.

### 2. Generate the dataset

Try the pipeline end to end first. This runs 2 tiny runs and finishes in minutes:
```bash
python cellparam/sim/generate.py --split train --out sim_data --smoke
```
Delete `sim_data/train` after the smoke test. The real train runs would otherwise skip the tiny smoke files, because they have the same names.

Then generate the four splits:
```bash
python cellparam/sim/generate.py --split train     --out sim_data   # 125 runs: Ka {0.1..0.5} x alpha {60..100} x seeds 0-4
python cellparam/sim/generate.py --split test      --out sim_data   #  32 runs: Ka {0.15..0.45} x alpha {65..95} x seeds 100-101
python cellparam/sim/generate.py --split extrap    --out sim_data   #   4 runs: Ka {0.05, 0.6}, alpha 80, seeds 200-201
python cellparam/sim/generate.py --split stability --out sim_data   #   1 run:  Ka 0.3, alpha 80, seed 300 (238 frames)
```
Each run has N=300 cells and 1500 frames, one frame every 5 minutes. Output goes to `sim_data/<split>/Ka<Ka>_alpha<alpha>_seed<seed>.npz`. Each train or test file is about 8 MB, so the full train split is about 1 GB.
`sim_data/` is in `.gitignore`, so every collaborator either generates the data themselves or gets a copy from someone who has.

Useful options:
- `--workers K` sets the number of parallel processes. The default is CPU cores − 1, so lower it if your machine needs to stay usable.
- `--N`, `--frames` change the tissue size and run length. Only change them on purpose: all runs in a dataset must use the same values.

**Resuming after an interruption.** If a run is interrupted (crash, shutdown, Ctrl+C), rerun **the same command**.
Each run is first written to a `.part.npz` file and renamed only when it is complete, so the files that already exist are always complete and get skipped.
If you see a leftover `*.part.npz` file, delete it. To count finished runs per split:
```bash
python -c "import os; [print(s, len([f for f in os.listdir(f'sim_data/{s}') if not f.endswith('.part.npz')])) for s in ['train','test','extrap','stability'] if os.path.isdir(f'sim_data/{s}')]"
```

Each `.npz` file contains these arrays:
- `pts` (frames × N × 2) cell centres
- `areas`, `erk`, `A0` (frames × N)
- `time`
- labels `Ka`, `alpha`, `seed`
- box size `L`, `frame_minutes`, and `A0_clamp_fraction` (how often the preferred-area floor was hit)
- `params` (all simulator parameters, as JSON)

### 3. Look at a run

```bash
python cellparam/sim/view.py sim_data/train/Ka0.10_alpha60_seed0.npz                        # frame 0, coloured by area
python cellparam/sim/view.py sim_data/train/Ka0.10_alpha60_seed0.npz --frame 500 --color erk
python cellparam/sim/view.py sim_data/train/Ka0.10_alpha60_seed0.npz --gif --frames 200      # animation
```
`--color` accepts `area`, `erk` or `A0`. The PNG or GIF is saved in the current directory.


## Installation

**PyAFV** is available on **PyPI** and can be installed using *pip* directly:
```bash
pip install pyafv
```
The package supports Python ≥ 3.10 and < 3.15, including Python 3.14t (the free-threaded, no-GIL build).
To verify that the installation was successful and that the correct version is installed, run the following in Python:
```python
import pyafv
print(pyafv.__version__)
```

<!--
> On some HPC clusters, global Python path can contaminate the runtime environment. You may need to clear it explicitly using `unset PYTHONPATH` or prefixing the *pip* command with `PYTHONPATH=""`.
-->

As an alternative, you can install **PyAFV** via *conda* from the **conda-forge** channel:
```bash
conda install -c conda-forge pyafv
```


<!--
### Install from source

Installing from source can be necessary if *pip* installation does not work. First, download and unzip the source code, then navigate to the root directory of the package and run:
```bash
pip install .
```

> **Note:** A C/C++ compiler is required if you are building from source, since some components of **PyAFV** are implemented in Cython for performance optimization.


#### Windows MinGW GCC

If you are using **MinGW GCC** (rather than **MSVC**) on *Windows*, to build from the source code, add a `setup.cfg` file at the repository root before running `pip install .` with the following content:
```ini
# setup.cfg
[build_ext]
compiler=mingw32
```


### Install offline

If you need to install **PyAFV** on a machine without internet access, you can download the corresponding wheel file from **PyPI** and transfer it to the target machine, and then run the following command to install using *pip*:
```bash
pip install pyafv-<version>-<platform>.whl
```
Alternatively, you can build **PyAFV** from source as described in the previous section. In this case, in addition to the required prerequisites of the package, the build-time dependencies **hatchling** and **hatch-cython** must also be available.
-->

> See the [documentation](https://pyafv.readthedocs.io/latest/usage.html#install-from-source) for instructions on installing the package from source or in offline environments.


### Install using Docker 🐳

**PyAFV** can also be installed via containerized environments. Pull the Docker image from **Docker Hub**:
```bash
docker pull wwang721/pyafv:latest
```
The image is also available via the **GitHub Container Registry (GHCR)** under [**GitHub Packages**](https://github.com/wwang721/pyafv/pkgs/container/pyafv); use `ghcr.io/wwang721/pyafv` to pull from GHCR instead.
<!--
Then run Python scripts with `pyafv` using:
```bash
docker run --rm -v $(pwd):/app wwang721/pyafv python <script_name>.py
```
Use `${PWD}` on Windows PowerShell instead of `$(pwd)`.
-->


## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pyafv/assets/blob/main/jupyter/getting_started.ipynb)

Here is a simple example to get you started, demonstrating how to construct a finite Voronoi diagram (click the **Google Colab** badge above to run the notebook directly):
```python
import numpy as np
import pyafv as afv

N = 100                                          # number of cells
pts = np.random.rand(N, 2) * 10                  # initial positions
params = afv.PhysicalParams(r=1.0)               # use default parameter values
sim = afv.FiniteVoronoiSimulator(pts, params)    # initialize the simulator
sim.plot_2d(show=True)                           # visualize the Voronoi diagram
```
To compute the conservative forces and extract detailed geometric information (e.g., cell areas, vertices, and edges), call:
```python
diag = sim.build()
```
The returned object `diag` is a Python `dict` containing these quantities. Refer to the [documentation](https://pyafv.readthedocs.io/latest/usage.html#a-simple-example) for more detailed usage.

## Simulation previews

Below are representative simulation snapshots generated using the code:

| Model illustration | Periodic boundary conditions |
|-----------------|-----------------|
| <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/model_illustration.png" width="540"> | <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/pbc.png" width="385">|

| Initial configuration | After relaxation | Active dynamics enabled |
|-----------------------|-----------------------|-----------------------|
| <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/initial_configuration.png" width="300"> | <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/relaxed_configuration.png" width="300"> | <img src="https://media.githubusercontent.com/media/wwang721/pyafv/main/assets/active_FV.png" width="300"> |


## More information

- **Full documentation** on [readthedocs](https://pyafv.readthedocs.io) or as [a single PDF file](https://pyafv.readthedocs.io/_/downloads/en/latest/pdf/).

- See [CONTRIBUTING.md](https://github.com/wwang721/pyafv/blob/main/.github/CONTRIBUTING.md) or the [documentation](https://pyafv.readthedocs.io/latest/contributing.html) for **local development instructions**.

- See some important [**issues**](https://github.com/wwang721/pyafv/issues?q=is%3Aissue%20state%3Aclosed) for additional context, such as: 
    * [QhullError when 3+ points are collinear #1](https://github.com/wwang721/pyafv/issues/1) [Closed - see [comments](https://github.com/wwang721/pyafv/issues/1#issuecomment-3701355742)]
    *  [Add customized plotting to examples illustrating access to vertices and edges #5](https://github.com/wwang721/pyafv/issues/5) [Completed in PR [#7](https://github.com/wwang721/pyafv/pull/7)]
    * [Time step dependence of intercellular adhesion in simulations #8](https://github.com/wwang721/pyafv/issues/8) [Closed in PR [#9](https://github.com/wwang721/pyafv/pull/9)]
    * [Handle degenerate 4-fold Voronoi vertices causing unpacking errors in _assemble_forces #38](https://github.com/wwang721/pyafv/issues/38) [Mentioned in PR [#42](https://github.com/wwang721/pyafv/pull/42) and PR [#58](https://github.com/wwang721/pyafv/pull/58)]

- Some releases of this repository are cross-listed on [Zenodo](https://doi.org/10.5281/zenodo.18091659):

  [![Zenodo](https://zenodo.org/badge/1124385738.svg)](https://doi.org/10.5281/zenodo.18091659)


## Citing the package

To cite **PyAFV**, please refer to the associated paper available on [arXiv](
https://doi.org/10.48550/arXiv.2604.15481). Here is a ready-made BibTeX entry:
```bibtex
@article{wang2026divergence,
  title   = {{Divergence of detachment forces in the finite Voronoi model}},
  author  = {Wang, Wei and Camley, Brian A},
  journal = {arXiv preprint arXiv:2604.15481},
  year    = {2026},
  doi     = {10.48550/arXiv.2604.15481}
}
```


## References

<table>
  <tr>
    <td id="huang2023bridging" valign="top">[1]</td>
    <td>
      J. Huang, H. Levine, and D. Bi, <em>Bridging the gap between collective motility and epithelial-mesenchymal transitions through the active finite Voronoi model</em>, <a href="https://doi.org/10.1039/D3SM00327B">Soft Matter <strong>19</strong>, 9389 (2023)</a>.
    </td>
  </tr>
  <tr>
    <td id="teomy2018confluent" valign="top">[2]</td>
    <td>
      E. Teomy, D. A. Kessler, and H. Levine, <em>Confluent and nonconfluent phases in a model of cell tissue</em>, <a href="https://doi.org/10.1103/PhysRevE.98.042418">Phys. Rev. E <strong>98</strong>, 042418 (2018)</a>.
    </td>
  </tr>
  <tr>
    <td id="wang2026divergence" valign="top">[3]</td>
    <td>
      W. Wang (汪巍) and B. A. Camley, <em>Divergence of detachment forces in the finite Voronoi model</em> <a href="https://doi.org/10.48550/arXiv.2604.15481">arXiv:2604.15481 [cond-mat.soft] (2026)</a>.
    </td>
  </tr>
</table>
