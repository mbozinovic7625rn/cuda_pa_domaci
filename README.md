# GPU Pollution-Diffusion Simulation — Parallel Algorithms (CUDA)

Second assignment for the *Paralelni algoritmi* (Parallel Algorithms) course at Računarski fakultet (RAF), Belgrade: a **CUDA/PyCUDA simulation of pollution spreading across a city grid**, with GPU-side analysis of dangerous zones.

## What it does

The simulation models pollutant concentration on a 2D grid over time:

- **`diffusion.cu`** — the core diffusion step: each cell's concentration evolves from its neighbors in parallel, one thread per cell.
- **`sources.cu`** — injection of pollution from active sources into the grid each step.
- **`danger.cu` / `ever_dangerous.cu`** — GPU reduction kernels counting cells currently above the danger threshold, and cells that have *ever* been dangerous during the simulation.
- **`region.cu`** — per-region hotspot statistics computed on the GPU.
- **`bonus_*.cu`** — bonus variants using per-region pollution profiles stored in GPU constant/global memory (`d_region_profiles`), with profiled region and danger statistics.

The Python host (`drugi_projekat.py`) compiles all kernels with PyCUDA (`SourceModule`, `sm_89`, fast-math), manages device memory, launches the simulation loop, and times GPU execution.

## Highlights

- Stencil computation (neighborhood diffusion) mapped to a CUDA thread grid
- Parallel reductions for counting and per-region statistics
- Use of GPU global symbols for region profiles in the bonus kernels
- Host↔device data management and kernel timing from Python

## Running it

Requires an NVIDIA GPU with the CUDA toolkit and PyCUDA:

```bash
pip install numpy pycuda
python drugi_projekat.py
```

(Kernels are compiled for `sm_89`; adjust `arch` in `drugi_projekat.py` for a different GPU.)

## Author

Mihailo Božinović — RN 76/2025, Računarski fakultet (RAF), Belgrade
