terraform {
  required_version = ">= 1.5"
  backend "azurerm" {
    resource_group_name  = "wine-tfstate-rg"
    storage_account_name = "winetfstate${random_string.suffix.result}"
    container_name       = "tfstate"
    key                  = "wine-mlops.tfstate"
  }
  required_providers {
    azurerm = { source = "hashicorp/azurerm" version = "~> 3.45" }
    random  = { source = "hashicorp/random"  version = "~> 3.6" }
  }
}

provider "azurerm" { features {} }

resource "random_string" "suffix" { length = 6 upper = false numeric = true special = false }
