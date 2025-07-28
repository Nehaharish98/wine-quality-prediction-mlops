.PHONY: help install test lint format clean build train deploy all

# Variables
PYTHON = python3
PIP = pip
PYTEST = pytest
BLACK = black
FLAKE8 = flake8
ISORT = isort
DOCKER = docker
DOCKER_COMPOSE = docker-compose

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	pre-commit install

test: ## Run tests
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

test-verbose: ## Run tests with verbose output
	$(PYTEST) tests/ -v --cov=src --cov-report=html --cov-report=term

lint: ## Run linting checks
	$(FLAKE8) src/ tests/ scripts/
	$(BLACK) --check src/ tests/ scripts/
	$(ISORT) --check-only src/ tests/ scripts/

format: ## Format code
	$(BLACK) src/ tests/ scripts/
	$(ISORT) src/ tests/ scripts/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf models/*.pkl
	rm -rf mlruns/
	rm -rf data/raw/*.csv
	rm -rf data/processed/*.parquet

data: ## Download and preprocess data
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/preprocess.py --in data/raw/wine.csv --out data/processed/train.parquet

train: ## Train model
	$(PYTHON) scripts/train_model.py --data data/processed/train.parquet --model models/model.pkl

dvc-repro: ## Run DVC pipeline
	dvc repro

build: ## Build Docker image
	$(DOCKER) build -t wine-pred:latest .

docker-run: ## Run Docker container locally
	$(DOCKER) run -p 8000:8000 wine-pred:latest

docker-compose-up: ## Start all services with docker-compose
	$(DOCKER_COMPOSE) up -d

docker-compose-down: ## Stop all services
	$(DOCKER_COMPOSE) down

monitoring-up: ## Start monitoring stack
	$(DOCKER_COMPOSE) -f monitoring/docker-compose.yml up -d

monitoring-down: ## Stop monitoring stack
	$(DOCKER_COMPOSE) -f monitoring/docker-compose.yml down

mlflow-up: ## Start MLflow server
	$(DOCKER_COMPOSE) -f mlflow-docker-compose.yml up -d

mlflow-down: ## Stop MLflow server
	$(DOCKER_COMPOSE) -f mlflow-docker-compose.yml down

prefect-start: ## Start Prefect server
	prefect server start

prefect-deploy: ## Deploy Prefect flows
	cd prefect_flows && $(PYTHON) deployment.py

deploy: build docker-compose-down docker-compose-up ## Build and deploy application

terraform-init: ## Initialize Terraform
	cd terraform && terraform init

terraform-plan: ## Plan Terraform deployment
	cd terraform && terraform plan

terraform-apply: ## Apply Terraform deployment
	cd terraform && terraform apply -auto-approve

terraform-destroy: ## Destroy Terraform resources
	cd terraform && terraform destroy -auto-approve

integration-test: ## Run integration tests
	$(PYTEST) tests/integration/ -v

api-test: ## Test API endpoints
	curl -f http://localhost:8000/health || echo "Health check failed"
	curl -f http://localhost:8000/metrics || echo "Metrics endpoint failed"

load-test: ## Run load tests (requires hey tool)
	@command -v hey >/dev/null 2>&1 || { echo "Please install hey: go install github.com/rakyll/hey@latest"; exit 1; }
	hey -n 100 -c 10 -m POST -H "Content-Type: application/json" \
		-d '{"fixed_acidity":7.4,"volatile_acidity":0.7,"citric_acid":0.0,"residual_sugar":1.9,"chlorides":0.076,"free_sulfur_dioxide":11.0,"total_sulfur_dioxide":34.0,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}' \
		http://localhost:8000/predict

security-scan: ## Run security scan
	bandit -r src/
	safety check

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

docs: ## Generate documentation
	@echo "API documentation available at http://localhost:8000/docs when server is running"

setup-dev: install data train ## Setup development environment

ci: lint test ## Run CI pipeline locally

full-pipeline: clean install lint test build docker-run api-test ## Run complete pipeline

all: clean install lint test build deploy monitoring-up api-test ## Run everything
	@echo "Wine Quality MLOps pipeline completed successfully!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "MLflow: http://localhost:5000"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"