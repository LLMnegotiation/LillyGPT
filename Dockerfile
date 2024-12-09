# Step 1: Use the official lightweight Python image as the base image
FROM python:3.9-slim

# Step 2: Set environment variables to prevent Python from writing .pyc files and enable output buffering
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Step 3: Set the working directory inside the container
WORKDIR /app

# Step 4: Copy all project files from the current directory to the container's working directory
COPY . .

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Download the SpaCy model (ensure it's available inside the container)
RUN python -m spacy download en_core_web_sm

# Step 7: Expose the port Gradio will run on (default is 7860)
EXPOSE 7860

# Step 8: Specify the command to run your app
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "app:demo"]
