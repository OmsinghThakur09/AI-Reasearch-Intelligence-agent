# stage 1: Builder
FROM python:3.12-slim AS builder

# grab the pre-compiled uv binary directly
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /build

# uv create the virtual env inside the /build directory
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# copy only the dependency files first
COPY pyproject.toml uv.lock ./

# installing depedencies into the .venv (skipping dev dependencies)
RUN uv sync --frozen --no-dev --no-install-project

# stage 2: final
FROM python:3.12-slim AS final

WORKDIR /app

# completely isolated virtual env from the builder
COPY --from=builder /app/.venv /app/.venv

# copy application code
COPY app/ ./app/

COPY config.py ./

EXPOSE 8000

# put the virtual environment in the execution path
ENV PATH="/app/.venv/bin:$PATH"

# run Uvicorn directy from the virtural environment
CMD ["uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
