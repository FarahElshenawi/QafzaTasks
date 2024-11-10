# Use official Python image as the base image
FROM python:3.9-slim

# Set environment variables to avoid writing .pyc files and to ensure output is logged to the terminal
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements_linux.txt ./

# Install dependencies
RUN pip install -r requirements_linux.txt

# Copy the rest of the project files into the container
COPY . /app/

# Expose port 8000 for FastAPI (Uvicorn)
EXPOSE 8000

# Command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "app.mlapi:app", "--host", "0.0.0.0", "--port", "8000"]
