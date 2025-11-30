# CrossCat

## Dockerized environments
Two container images capture the environments described in `TODO.md`.

| Target | Dockerfile | Image name (via Compose) | Notes |
| --- | --- | --- | --- |
| Modern GenJAX benchmarks | `docker/genjax/Dockerfile` | `crosscat-genjax` | CUDA 12.4 runtime, Python 3.11 managed via `uv`, installs the repo with the locked dependencies. |
| Legacy probcomp/crosscat | `docker/legacy/Dockerfile` | `crosscat-legacy` | Ubuntu 18.04 base, Python 2.7 toolchain pinned to values tolerated by the historical C++/Python backend. |

Both containers mount the repository inside `/workspaces/crosscat`, so results, datasets (`./data`, `./experiment`, etc.), and configuration files are automatically shared with the host.

### Quick start (Docker Compose)
```bash
# Build both images (runs uv/crosscat installs once)
docker compose -f docker/docker-compose.yml build

# Start a GPU-enabled GenJAX shell
docker compose -f docker/docker-compose.yml run --rm --gpus all genjax

# Start the legacy probcomp/crosscat shell
docker compose -f docker/docker-compose.yml run --rm legacy
```

Inside the GenJAX container the Python environment lives in `/opt/venv` and `uv` is available. Typical commands:

```bash
uv run pytest
uv run python -m crosscat.main
# Reproduce the legacy DHA inference demo with the GenJAX sampler
uv run python examples/dha_genjax.py data/dha.csv --iterations 200 --num-views 5 --num-clusters 8
```
These commands assume `PYTHONPATH` points at `./src` (the Docker images export it as `/workspaces/crosscat/src`). When running directly on the host, either set `PYTHONPATH=src` or `uv pip install -e .` once.

Inside the legacy container the editable checkout of `probcomp/crosscat` is under `/opt/probcomp/crosscat` and relies on Python 2.7-era binaries. Example usage:

```bash
python2.7 -m crosscat.tests.unit_tests.test_continuous_component_model
python2.7 scripts/crosscat_cli.py --help
```

### DHA demo: GenJAX vs legacy CrossCat
You can reproduce the DHA inference example with both samplers and compare their diagnostics:

1. Run the modern GenJAX script directly on the host (or inside the `genjax` container):
   ```bash
   PYTHONPATH=src uv run python examples/dha_genjax.py data/dha.csv \
     --max-rows 50 \
     --iterations 100 \
     --predictive-samples 10 \
     --impute-samples 200 \
     --num-views 5 \
     --num-clusters 8
   ```
   Review the predictive checks and imputations it prints to verify behavior against the legacy sampler.

2. Run the historical Python 2.7 implementation inside the legacy container:
   ```bash
   docker compose -f docker/docker-compose.yml run --rm legacy bash -lc \
     "cd /opt/probcomp/crosscat && python2.7 examples/dha_example.py www/data/dha.csv \
       --num_chains 2 \
       --num_transitions 100"
   ```
   The legacy script emits the same reconstruction rows and imputations for easy side-by-side comparison.

### Manual builds
You can build and run the images without Compose, e.g.

```bash
docker build -f docker/genjax/Dockerfile -t crosscat-genjax .
docker run --rm -it --gpus all -v "$(pwd)":/workspaces/crosscat crosscat-genjax bash
```

Compose simply codifies the recommended mount points and cache volume (`genjax-uv-cache`) for faster rebuilds.
