#!/bin/bash

# Print a message
echo "Listing the contents of the 'models' directory before building the Docker image"


# List the contents of the 'models' directory
ls -lh models

# Build the Docker image with the specified tag and Dockerfile
docker build -t digits:v1 -f docker/DockerFile .

# Run the Docker container with the specified volume binding
docker run -it -v /mnt/c/Users/Sourabh\ Chawda/IITJ/MLOps/digit-classification/models:/digits/models digits:v1

# Print a message
echo "Listing the contents of the 'models' directory after running the Docker container"

# List the contents of the 'models' directory again
ls -lh models
