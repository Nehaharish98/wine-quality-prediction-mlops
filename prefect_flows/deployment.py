"""Prefect deployment configuration."""
from prefect import flow
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.infrastructure.docker import DockerContainer
from training_pipeline import training_pipeline

# Create deployment with schedule
deployment = Deployment.build_from_flow(
    flow=training_pipeline,
    name="wine-quality-training-scheduled",
    parameters={},
    schedule=CronSchedule(cron="0 2 * * *"),  # Daily at 2 AM
    work_queue_name="wine-quality-training",
    infrastructure=DockerContainer(
        image="wine-pred:latest",
        image_pull_policy="Always",
        networks=["wine-network"]
    ),
    tags=["wine-quality", "training", "mlops"]
)

if __name__ == "__main__":
    deployment.apply()
    print("Deployment created successfully!")
