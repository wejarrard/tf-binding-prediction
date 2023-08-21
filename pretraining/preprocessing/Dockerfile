# To build the image: docker build -t preprocessing .
# To ssh into the container: docker run -it preprocessing /bin/bash
# Stop all containers: docker stop $(docker ps -a -q)
# Stop just preprocessing container: docker stop $(docker ps -a | grep preprocessing | awk '{print $1}')

# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.10-slim as build

# Set the working directory in the container
# WORKDIR /app

# Copy the current directory contents into the container at /
ADD requirements.txt /

# Install additional system packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    zlib1g-dev \
    bedtools \
    samtools \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt