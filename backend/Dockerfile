# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port for the application
EXPOSE 8080

# Define environment variable for Flask to run in production
ENV FLASK_ENV=production

# Run the application
CMD ["python", "backend.py"]
