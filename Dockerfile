# Use the custom base image instead of python:3.9-slim
FROM docker-registry.dev.displayme.net/linux_base

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if necessary for your environment)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Ensure that pip is linked to Python 3, only if not already linked
RUN [ ! -e /usr/bin/python ] && ln -s /usr/bin/python3 /usr/bin/python || echo "Python symlink already exists"
RUN [ ! -e /usr/bin/pip ] && ln -s /usr/bin/pip3 /usr/bin/pip || echo "Pip symlink already exists"

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
