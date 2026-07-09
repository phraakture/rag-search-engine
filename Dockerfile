FROM python:3.14-slim

WORKDIR /app

COPY cli/ cli/
COPY data/ data/
COPY cache/ cache/
COPY pyproject.toml .

RUN pip install --no-cache-dir \
    nltk \
    numpy \
    openai \
    pillow \
    python-dotenv \
    sentence-transformers

ENV PYTHONPATH=/app/cli

ENTRYPOINT ["python"]
CMD ["cli/hybrid_search_cli.py", "--help"]
