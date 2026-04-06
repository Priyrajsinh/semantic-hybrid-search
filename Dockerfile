# Stage 1 — install dependencies into a clean prefix
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Stage 2 — minimal runtime image
FROM python:3.10-slim AS runtime

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy only what the server needs at runtime
COPY src/    ./src/
COPY models/ ./models/
COPY config/ ./config/
COPY reports/results.json ./reports/results.json

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
