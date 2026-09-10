FROM python:3.12-slim AS build
WORKDIR /build

# Dependencies before source: they change far less often, so a code edit reuses the cached
# layer instead of reinstalling numpy and fastapi every time.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
WORKDIR /app

COPY --from=build /install /usr/local

RUN useradd --system --create-home --uid 10001 server
USER server

EXPOSE 8000
ENTRYPOINT ["python", "-m", "inference_server.server", "--host", "0.0.0.0", "--port", "8000"]
