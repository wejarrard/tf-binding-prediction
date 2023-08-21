# To build the image: docker build -t tf-footprinting .
# To ssh into the container: docker run -it tf-footprinting /bin/bash
# Stop all containers: docker stop $(docker ps -a -q)
# Stop just tf-footprinting container: docker stop $(docker ps -a | grep tf-footprinting | awk '{print $1}')

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /
ADD tf-footprinting.sh .

# Copy data from your absolute path into the container
COPY ./data/ /app/data/

# Install additional system packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    zlib1g-dev \
    bedtools \
    samtools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install tobias

# Set the script to be executable
RUN chmod +x /app/tf-footprinting.sh

# Set the entrypoint or command to run the script
# ENTRYPOINT ["/app/tf-footprinting.sh"]