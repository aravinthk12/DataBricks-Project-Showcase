# House Price Prediction with Spark

## Overview
This project demonstrates a simple machine learning pipeline using PySpark and Databricks for data processing, model training, and deployment. The project allows you to run and test four different models based on runtime arguments. It is designed to be run locally with support for Databricks as the primary data processing platform.

The code is built with a focus on Spark for distributed data processing and includes Databricks Asset Bundles for deployment. While you can test the code locally, the primary orchestration, including scheduling and deployment, is done using Databricks workflows.

The project is under development, and certain features like MLflow integration with Databricks are still being worked on.


## Key Features
- **Data Processing**: Built on PySpark, allowing distributed data handling for large datasets.
- **Model Training**: The project supports training and testing of four different machine learning models based on runtime arguments.
- **MLflow Integration**: MLflow is used for experiment tracking. While it's currently configured for local use, integration with Databricks is in progress.
- **Databricks**: This project is designed to leverage Databricks as the main platform for orchestrating data processing and model training. It includes a Databricks asset bundle for easy deployment.

## Requirements
- Python 3.10+
- Poetry for dependency management
- Databricks CLI v0.227.0
## Setup Instructions

### 1. Download Python
To check if python is available, open command prompt and type:
```bash
python --version
```

### 2. Clone the repository
```bash
git clone https://github.com/aravinthk12/DataBricks-Project-Showcase
cd DataBricks-Project-Showcase
```
### 3. Install Dependencies
```bash
conda create -n show_venv python=3.10
pip install poetry
poetry install # to install all the required dependency
```
Note: 
1. I prefer conda for environment management, you can use your preferred environment manager
2. Use the EntryPoint.py to test run these codes
3. The main_notebook.ipynb exists for Databricks deployment

#### The above steps should be enough to run this project locally, but if you want to test this in Databricks, you would need to follow the below steps

### 4. Databricks cli
Please follow the steps on official documentation: [Link](https://docs.databricks.com/en/dev-tools/cli/install.html)
To check if databricks cli is available, open command prompt and type:
You can also get it from [GitHub - databricks/cli](https://github.com/databricks/cli/releases/tag/v0.227.0)
and configure it in your env variables
```bash
databricks --version 
```
Note: This step is optional as this is required for folks who are planning to deploy bundles in databricks

### 5. Configure Databricks
Configure the CLI with your Databricks workspace by generating a token from your Databricks account:
```bash
databricks configure 
```
Enter the following details:\
Host: Your Databricks workspace URL (e.g., https://<databricks-instance>.cloud.databricks.com) \
Token: The token you generated in the Databricks console.

### 6. MlFlow
This project includes MLflow for experiment tracking. You can run MLflow locally to log metrics and models during training.
To start MLflow:
```bash
mlflow ui
```
Then open http://127.0.0.1:5000 to view the MLflow dashboard.

Note: MLflow integration with Databricks is still a work in progress. Stay tuned for updates!

## Databricks Integration
As mentioned earlier, this project is designed to work seamlessly with Databricks for orchestration and deployment. 

## To deploy the project to Databricks, follow the steps below:
For more information on Databricks asset Bundles, check out [Link](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/bundles/)

### 1. Validate the Bundle
```bash
databricks bundle validate
```
### 2. Deploy the Bundle
```bash
databricks bundle deploy -t dev # to deploy the development
```

### 3. Run the workflow
```bash
databricks bundle run HousePricePrediction
```

## Future Work

- Full integration of MLflow with Databricks for experiment tracking.
- CI/CD pipelines for automating Databricks deployment.
- Additional models and hyperparameter tuning features.
- More MLFlow capabilities (As that was main reason for me start working on this project)
- Hydra config management for models

## Contributing
Feel free to fork this repository and submit pull requests if you'd like to contribute to this project. Any feedback or suggestions are welcome!

