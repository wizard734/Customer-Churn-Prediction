FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure the 'models' directory exists
RUN mkdir -p models

# Create a non-root user for running the application
RUN useradd -m appuser
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Command to run when the container starts
CMD ["python", "main.py", "--api", "--port", "5000"] 