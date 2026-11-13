# Exercise 3 course container: Streamlit app + Julia/Wflow + Claude Agent SDK + Mistral Vibe.
# The code-writing agents (A4, A4m, A5, harness nodes) run here instead of on participants' laptops.
#
# Build context = repo root (the .env there is baked in; delete the resource group after the course):
#   az acr build -r <registry> -t hydro-ex2:latest -f exercises/03_agentic_workflows/deploy/Dockerfile .

FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Everything runs as an unprivileged user; the agents get this user's rights and nothing more.
RUN useradd --create-home --uid 1000 app && mkdir -p /home/app/repo && chown app:app /home/app/repo
USER app
WORKDIR /home/app/repo
ENV PATH="/home/app/repo/.venv/bin:/home/app/.local/bin:/home/app/.juliaup/bin:$PATH"

# Julia + Wflow 1.x, precompiled (slow layer, cached unless this block changes).
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes \
 && julia -e 'using Pkg; Pkg.add(name="Wflow", version="1"); using Wflow'

# Mistral Vibe (A4m harness).
RUN uv tool install mistral-vibe

# Python deps from the lockfile, minus dllmforge/docling (Exercise 1 only; drags in CUDA torch).
COPY --chown=app pyproject.toml uv.lock .python-version ./
RUN uv export --frozen --no-dev --no-emit-project --no-hashes --prune dllmforge -o /tmp/req.txt \
 && uv venv .venv \
 && uv pip install --python .venv/bin/python -r /tmp/req.txt \
 && rm /tmp/req.txt

# Repo (filtered by .dockerignore at the repo root), including .env.
COPY --chown=app . .
# A context archived on Windows can arrive with folders app cannot write; fix ownership (skip the large .venv).
USER root
RUN find . -path ./.venv -prune -o -exec chown app:app {} + -exec chmod u+rwX {} +
USER app

# Baseline run + twin "observations", so the app starts ready.
RUN python exercises/03_agentic_workflows/cli.py prepare

# Base model read-only for the agents (owned by root, no write bits).
USER root
RUN chown -R root:root wflow_moselle_epinal && chmod -R a-w wflow_moselle_epinal
USER app

ENV HYDROAGENT_ALLOW_CODE_EXEC=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl --fail http://localhost:8000/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "exercises/03_agentic_workflows/app/streamlit_app.py", \
            "--server.port=8000", "--server.address=0.0.0.0", "--server.headless=true", \
            "--server.fileWatcherType=none", "--browser.gatherUsageStats=false"]
