variable "resource_group_name" {
  description = "Name of the resource group"
  type        = string
  default     = "mlops-wine-rg"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "West US 2"
}

variable "container_registry_name" {
  description = "Name of the container registry"
  type        = string
  default     = "winecr"
}

variable "container_app_env_name" {
  description = "Name of the container app environment"
  type        = string
  default     = "wine-ca-env"
}

variable "container_app_name" {
  description = "Name of the container app"
  type        = string
  default     = "wine-api"
}

variable "log_analytics_workspace_name" {
  description = "Name of the Log Analytics workspace"
  type        = string
  default     = "wine-logs"
}