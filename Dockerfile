# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Upgrade pip to the latest version
RUN python -m pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies using the latest pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on (use the port from the environment variable or default to 8000)
EXPOSE 8000

# Set environment variables (You can override these when running the container)
ENV HOST=0.0.0.0
ENV PORT=8000

# Command to run the application
CMD ["python3", "app.py"]
