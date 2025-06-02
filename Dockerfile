# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install -r requirements.txt

# Copy the entire project
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Create entrypoint script
RUN echo '#!/bin/bash\nset -e\nexec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"] 