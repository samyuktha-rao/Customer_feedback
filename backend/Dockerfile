# Example Dockerfile for Backend (Python Flask/FastAPI)
# ✅ Use a Python base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy only requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
