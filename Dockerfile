# Use an official PyTorch image as the base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /app

# Copy all contents to the container
COPY requirements.txt /app

# Set the environment variable
ENV WANDB_API_KEY=${WANDB_API_KEY}

# Install git
RUN apt-get update && apt-get install -y git

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
