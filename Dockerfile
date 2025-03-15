# Stage 1: Build dependencies separately for caching
FROM python:3.10-slim AS builder
WORKDIR /app

# Install dependencies separately for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Copy only necessary files to the final image
FROM python:3.10-slim
WORKDIR /app

# Install curl in the final image (needed for healthcheck)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy dependencies from builder stage (both site-packages and binaries)
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app files
COPY . .

EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Entry point to run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


## Stage 1: Build dependencies separately for caching
# FROM python:3.10-slim AS builder
# WORKDIR /app

# # Install dependencies separately for better caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Stage 2: Copy only necessary files to the final image
# FROM python:3.10-slim
# WORKDIR /app

# # Copy dependencies from builder stage (both site-packages and binaries)
# COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
# COPY --from=builder /usr/local/bin /usr/local/bin

# # Copy app files
# COPY . .
# RUN apt-get update && apt-get install -y curl
# EXPOSE 8501

# # Health check (optional)
# HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# # Entry point to run Streamlit
# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]




# FROM python:3.10-slim

# WORKDIR /app

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .
# # COPY data /app/data


# EXPOSE 8501

# HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]