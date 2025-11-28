FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# --- Base tooling ---------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        python3 \
        python3-pip \
        python3-venv \
        pkg-config \
        libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

# --- uv & Python toolchain ------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
RUN uv python install 3.11
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspaces/crosscat

# --- Locked dependency sync ----------------------------------------------
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --python 3.11

# --- Project files -------------------------------------------------------
COPY . .

ENV PYTHONPATH=/workspaces/crosscat/src
CMD ["bash"]
