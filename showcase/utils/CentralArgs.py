from dataclasses import dataclass
from pyspark.sql import SparkSession
from datetime import datetime as dt


@dataclass
class CentralArgs:
    spark: SparkSession
    process_date: str = dt.now().strftime("%Y-%m-%d")
    ModelName: str = None
    FeatureList: list = None
    environment: str = "dev"
    mlflow_login: str = "false"
