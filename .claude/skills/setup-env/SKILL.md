---
name: setup-env
description: Interactively onboard contributors to this project's Python environment, choose CPU or NVIDIA CUDA dependencies, install them, and verify the setup. Use for initial dependency setup or environment repair; use run-sim for running an already configured simulation.
---

# Set up the project environment

Work from the repository root. Read `requirements.txt` and, on macOS,
`scripts/build_pybullet_macos.sh` before selecting installation commands. This
skill owns the setup procedure; `requirements.txt` owns shared dependency pins.
Keep the PyTorch commands below aligned with that file when pins change.

## Inspect first

Detect OS and architecture, available Python interpreters, and whether `.venv`
exists. If present, inspect its Python version and `python -m pip check`.
Check `nvidia-smi` on Linux when available. Do not ask users to report facts you
can inspect. Do not print environment variables or credentials.

Recommend Python 3.11 for this older pinned stack; it is a compatibility baseline,
not an exclusive requirement. Do not infer full-stack compatibility from passing
physics tests alone. If a different interpreter is requested, verify wheel and
dependency availability before installing. If the required interpreter or macOS
Command Line Tools are missing, explain the prerequisite and ask which available
installation method the user prefers before changing system tooling.

## Ask only unresolved questions

Use the host's interactive question tool when available, otherwise ask in chat.
Honor choices already supplied in the request. Bundle independent questions and
wait for answers before doing work that depends on them.

- On a supported Linux NVIDIA machine: “Use CPU-only dependencies (smaller install)
  or NVIDIA CUDA dependencies (GPU training)?” Recommend CUDA when the user wants
  GPU training. If hardware/driver support is uncertain, offer CPU or a driver
  diagnosis; do not silently choose CUDA.
- On macOS or a machine without a supported NVIDIA GPU, explain that the
  CPU path below applies; skip a hardware-choice question with only one viable answer.
- If an existing environment needs changes: explain what is installed and ask
  whether to repair it in place or preserve it and create a fresh `.venv`.
  Do not remove or overwrite an existing environment without authorization.

Keep the setup to the full shared requirements by default. Do not offer a minimal
install unless the user requests it. Explain that PyBullet and CMA-ES run on the
CPU even with CUDA PyTorch; Apple GPUs do not use the CUDA path.

## Install and verify

Once choices are resolved, carry out the selected commands below. Use the venv's
Python explicitly for every pip invocation; no shell activation or global pip.
Install the selected PyTorch build before the shared requirements. On macOS,
run the patched PyBullet build after the shared requirements. Preserve existing
files when creating a fresh environment, for example by moving the previous venv
to an unused backup path after the user selects that option.

Do not upgrade project pins to solve an installation failure without discussing
the concrete incompatibility. Diagnose errors before retrying; report unresolved
prerequisites rather than claiming completion. Follow the host's permission flow
when downloads or protected writes require it.

Run the pip consistency check, dependency import smoke check, and
headless tipping tests. For CUDA, also verify `torch.cuda.is_available()` and
report the detected GPU. Do not open the GUI as part of onboarding unless asked.

Finish with the interpreter, CPU/CUDA choice, checks that passed, any unresolved
failure, and the command to run the environment demo. Distinguish a successful
installation from a workflow that has only been written or inspected.

## Installation commands

**Python 3.11 is recommended** as a compatibility baseline for the pinned stack.
Python 3.13/3.14 are unsuitable for a fresh install of these older pins.
Commands below assume Linux or macOS and the repo root. No venv activation is
needed. Start with a fresh environment when switching between CPU and CUDA.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
```

Choose **one** of the following setup paths. Both install the full scientific,
notebook, RL, simulation, and SysID stack from `requirements.txt`.

### Without a GPU (CPU)

**Linux:** install the CPU-only PyTorch wheel first, then the shared dependencies.

```bash
.venv/bin/python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

**macOS:** use the standard PyTorch wheel and build patched PyBullet from source.
The build requires Apple's Command Line Tools (`xcode-select --install` if missing)
and can take several minutes.

```bash
.venv/bin/python -m pip install torch==2.3.1
.venv/bin/python -m pip install -r requirements.txt
bash scripts/build_pybullet_macos.sh
```

macOS skips PyBullet in `requirements.txt`; the build script installs the same
pinned version with the Apple SDK zlib fix. This path also applies to Macs with
Apple GPUs; CUDA is unavailable on macOS. Use `device="cpu"` in native PyTorch SAC
training code when CPU execution is intended.

### With an NVIDIA GPU (CUDA)

For **Linux x86-64 with a CUDA 12.1-compatible NVIDIA driver**. Check that
`nvidia-smi` works before installing. Other GPU vendors should use the CPU path
for this pinned stack.

```bash
.venv/bin/python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; check the driver and installed wheel"; print(torch.cuda.get_device_name(0))'
```

The wheel choices follow the [official PyTorch 2.3.1 installation instructions](https://docs.pytorch.org/get-started/previous-versions/#v231).
The shared `torch==2.3.1` pin accepts either installed build. GPU acceleration
applies to neural-network training; the current PyBullet simulation and CMA-ES
studies run on the CPU.

### Verify either setup

```bash
.venv/bin/python -m pip check
.venv/bin/python -c 'import cma, gymnasium, matplotlib, numpy, pandas, pybullet, scipy, torch; print("Dependencies OK; torch:", torch.__version__)'
.venv/bin/python -m src.simulation.tools.validation.tipping_tests
```

`requirements.txt` is the source of truth for shared dependency versions,
including `cma` for M2. It pins direct dependencies, rather than every transitive
dependency. No separate package installs are needed for the SysID scripts.
