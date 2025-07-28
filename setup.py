"""Setup configuration for wine quality prediction package."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wine-quality-prediction",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="MLOps pipeline for wine quality prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/wine-quality-prediction-mlops",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
        ],
        "cloud": [
            "azure-storage-blob>=12.19.0",
            "azure-identity>=1.15.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "wine-predict=wine_quality.api.app:main",
            "wine-train=scripts.train_model:main",
        ],
    },
    include_package_data=True,
    package_data={
        "wine_quality": ["*.yaml", "*.yml", "*.json"],
    },
)