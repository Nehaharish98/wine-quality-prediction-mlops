#!/bin/bash
set -e

echo "=== Wine Quality Model Deployment Script ==="

# Configuration
ACR_NAME="winecr"
IMAGE_NAME="wine-pred"
CONTAINER_APP_NAME="wine-api"
RESOURCE_GROUP="mlops-wine-rg"

# Check if logged into Azure
if ! az account show > /dev/null 2>&1; then
    echo "Please log in to Azure CLI first: az login"
    exit 1
fi

# Get current timestamp for tagging
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
IMAGE_TAG="${TIMESTAMP}"

echo "Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest

echo "Logging into Azure Container Registry..."
az acr login --name ${ACR_NAME}

echo "Tagging image for ACR..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:latest

echo "Pushing image to ACR..."
docker push ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}
docker push ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:latest

echo "Getting ACR credentials..."
ACR_SERVER=$(az acr show --name ${ACR_NAME} --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name ${ACR_NAME} --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name ${ACR_NAME} --query passwords[0].value --output tsv)

echo "Updating Container App..."
az containerapp update \
    --name ${CONTAINER_APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --image ${ACR_SERVER}/${IMAGE_NAME}:${IMAGE_TAG} \
    --registry-server ${ACR_SERVER} \
    --registry-username ${ACR_USERNAME} \
    --registry-password ${ACR_PASSWORD}

# Get the app URL
APP_URL=$(az containerapp show \
    --name ${CONTAINER_APP_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --query properties.configuration.ingress.fqdn \
    --output tsv)

echo "=== Deployment Complete! ==="
echo "App URL: https://${APP_URL}"
echo "Health Check: https://${APP_URL}/health"
echo "API Docs: https://${APP_URL}/docs"
echo "Image: ${ACR_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"

# Test the deployment
echo "Testing deployment..."
sleep 10
curl -f "https://${APP_URL}/health" || echo "Health check failed"

echo "Deployment script completed successfully!"