#!/bin/bash

# Build the Docker image
docker build -f aspect/Dockerfile -t aspect-docker .

# Run the container with JupyterLab on port 8888
docker run --rm --name aspect-docker -d -p 8888:8888 aspect-docker

# Access this container by going to http://localhost:8888 in your browser.
# Stop this container by running "docker stop aspect-docker"