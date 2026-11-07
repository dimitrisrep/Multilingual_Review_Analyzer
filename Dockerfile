FROM python:3.12-slim
WORKDIR /app
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
USER 65534:65534
EXPOSE 8002
CMD ["uvicorn", "review_analyzer.api:app", "--host", "0.0.0.0", "--port", "8002"]
