FROM python:3.12-slim

# Install as root so the streamlit script lands in /usr/local/bin (on PATH)
COPY requirements.txt /tmp/requirements.txt
RUN PIP_DEFAULT_TIMEOUT=120 pip install --no-cache-dir -r /tmp/requirements.txt

# Non-root user with a home dir: streamlit writes ~/.streamlit
RUN useradd --create-home --uid 1000 app
USER app
WORKDIR /app

# App code comes from a volume mount (live edits), not from the image
CMD ["streamlit", "run", "/app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
