terraform {
  required_version = ">=1.5"
  required_providers { azurerm = { source = "hashicorp/azurerm" version = "~>3.60" } }

  backend "local" {}                    # keep state file on disk; S3/GCS cost $$
}

provider "azurerm" { features {} }

resource "azurerm_resource_group" "rg" {
  name     = var.rg_name
  location = var.region
}

# ✨  MLflow on Azure Container Apps – EDU-friendly pay-per-second
resource "azurerm_container_app_environment" "cae" {
  name                 = "wine-cae"
  location             = azurerm_resource_group.rg.location
  resource_group_name  = azurerm_resource_group.rg.name
}

resource "azurerm_container_app" "mlflow" {
  name                       = "mlflow"
  container_app_environment_id = azurerm_container_app_environment.cae.id
  resource_group_name        = azurerm_resource_group.rg.name
  revision_mode              = "Single"

  template {
    container {
      name   = "mlflow"
      image  = "ghcr.io/mlflow/mlflow:latest"
      cpu    = 0.25                     # smallest EDU-legal size
      memory = "0.5Gi"

      env {
        name  = "MLFLOW_BACKEND_STORE_URI"
        value = "sqlite:///mlflow.db"
      }
      env {
        name  = "MLFLOW_ARTIFACT_ROOT"
        value = "file:/artifacts"       # local volume → free
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 5000
    traffic_weight {
      percentage = 100
      latest_revision = true
    }
  }
}

output "mlflow_url" { value = azurerm_container_app.mlflow.latest_revision_fqdn }