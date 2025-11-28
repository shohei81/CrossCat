# CrossCat

## Dockerized environments
Two container images capture the environments described in `TODO.md`.

| Target | Dockerfile | Image name (via Compose) | Notes |
| --- | --- | --- | --- |
| Modern GenJAX benchmarks | `docker/genjax/Dockerfile` | `crosscat-genjax` | CUDA 12.4 runtime, Python 3.11 managed via `uv`, installs the repo with the locked dependencies. |
| Legacy probcomp/crosscat | `docker/legacy/Dockerfile` | `crosscat-legacy` | Ubuntu 18.04 base, Python 3.8 toolchain pinned to values tolerated by the historical C++/Python backend. |

Both containers mount the repository inside `/workspaces/crosscat`, so results, datasets (`./data`, `./experiment`, etc.), and configuration files are automatically shared with the host.

### Quick start (Docker Compose)
```bash
# Build both images (runs uv/crosscat installs once)
docker compose -f docker/docker-compose.yml build

# Start a GPU-enabled GenJAX shell
docker compose -f docker/docker-compose.yml run --rm --gpus all genjax

# Start the legacy probcomp/crosscat shell
docker compose -f docker/docker-compose.yml run --rm probcomp
```

Inside the GenJAX container the Python environment lives in `/opt/venv` and `uv` is available. Typical commands:

```bash
uv run pytest
uv run python -m crosscat.main
```

Inside the legacy container the editable checkout of `probcomp/crosscat` is under `/opt/probcomp/crosscat` and relies on Python 2.7-era binaries. Example usage:

```bash
python2.7 -m crosscat.tests.test_timing
python2.7 scripts/crosscat_cli.py --help
```

### Manual builds
You can build and run the images without Compose, e.g.

```bash
docker build -f docker/genjax/Dockerfile -t crosscat-genjax .
docker run --rm -it --gpus all -v "$(pwd)":/workspaces/crosscat crosscat-genjax bash
```

Compose simply codifies the recommended mount points and cache volume (`genjax-uv-cache`) for faster rebuilds.
