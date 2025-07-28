#!/bin/bash
set -e

echo "Building Docker image..."
docker build -t wine-pred:latest .

echo "Tagging for Azure Container Registry..."
docker tag wine-pred:latest winecr.azurecr.io/wine-pred:$(date +%Y%m%d-%H%M%S)

echo "Pushing to registry..."
az acr login --name winecr
docker push winecr.azurecr.io/wine-pred:$(date +%Y%m%d-%H%M%S)

echo "Deployment complete!"