terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    azapi = {
      source  = "Azure/azapi"
      version = "~> 1.0"
    }
  }
  backend "azurerm" {
    resource_group_name  = "tfstate-rg"
    storage_account_name = "tfstatewinepred"
    container_name       = "tfstate"
    key                  = "wine-pred.tfstate"
  }
}

provider "azurerm" {
  features {}
}

provider "azapi" {}
