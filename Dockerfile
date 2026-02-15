FROM python:3.9-slim

WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# --- THE MAGIC FIX ---
# This forces Streamlit to print "http://localhost:8501" instead of "0.0.0.0"
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "[browser]\nserverAddress = \"localhost\"\nserverPort = 8501" > /root/.streamlit/config.toml'
# ---------------------

# Copy the app code
COPY . .

# Expose the port
EXPOSE 8501

# Run the app
# Note: We still bind to 0.0.0.0 internally so Docker works, but the config above fixes the display.
ENTRYPOINT ["streamlit", "run", "alpha_engine.py", "--server.port=8501", "--server.address=0.0.0.0"]