# Use Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Train model
RUN python model_training.py

# Expose port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]