# syntax=docker/dockerfile:1

# ---- 1. Build the SvelteKit SPA ----
FROM node:24-slim AS frontend
WORKDIR /app/frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml frontend/.npmrc ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

# ---- 2. Python base with uv ----
FROM python:3.14-slim AS python-base
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# ---- Dev target (docker compose --profile dev): source is bind-mounted ----
FROM python-base AS backend-dev
WORKDIR /app/backend
CMD ["sh", "-c", "uv sync --frozen && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]

# ---- 3. Runtime ----
FROM python-base AS runtime
WORKDIR /app
COPY backend/pyproject.toml backend/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project
COPY backend/app ./app
COPY backend/weights ./weights
COPY --from=frontend /app/frontend/build ./static

RUN useradd --create-home --uid 1000 app
USER app

ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/api/health')"
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
