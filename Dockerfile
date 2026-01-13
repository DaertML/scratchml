FROM node:20-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ripgrep \
    curl \
    nano \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Claude tools globally (npm is fine)
RUN npm install -g @anthropic-ai/claude-code @musistudio/claude-code-router

# --- PYTHON FIX START ---
# Create a virtual environment for Python tools
RUN python3 -m venv /opt/venv
# Ensure the node user can use it
RUN chown -R node:node /opt/venv

# Add the venv to the PATH so 'pytest', 'flake8', etc., work automatically
ENV PATH="/opt/venv/bin:$PATH"

# Switch to the node user BEFORE installing python tools
RUN mkdir -p /project && chown -R node:node /project
WORKDIR /project
USER node

# Install Python tools into the venv (no longer externally managed)
RUN pip3 install --no-cache-dir \
    pytest \
    flake8 \
    mypy
# --- PYTHON FIX END ---

EXPOSE 3456

CMD ["sh", "-c", "ccr start && tail -f /dev/null"]
