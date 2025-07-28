resource "azurerm_container_group" "mlflow" {
  name                = "mlflow-server"
  location            = azurerm_resource_group.core.location
  resource_group_name = azurerm_resource_group.core.name
  os_type             = "Linux"

  container {
    name   = "mlflow"
    image  = "ghcr.io/mlflow/mlflow:latest"
    cpu    = "1"
    memory = "2"

    ports { port = 5000 }

    environment_variables = {
      MLFLOW_BACKEND_STORE_URI     = "sqlite:///mlflow.db"
      MLFLOW_ARTIFACT_ROOT         = "wasbs://${azurerm_storage_container.mlflow_container.name}@${azurerm_storage_account.mlflow_sa.name}.blob.core.windows.net/"
      AZURE_STORAGE_CONNECTION_STRING = azurerm_storage_account.mlflow_sa.primary_connection_string
    }
  }

  ip_address_type = "Public"
  dns_name_label  = "wine-mlflow-${random_string.suffix.result}"
}
output "mlflow_url" { value = "http://${azurerm_container_group.mlflow.fqdn}:5000" }
