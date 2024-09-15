from showcase.HousePricePrediction.data_preprocessing.DataPreProcessingHousePricePredictor import (
    DataPreProcessingHousePricePredictor,
)
from showcase.HousePricePrediction.model.HousePricePredictor import (
    HousePricePredictor,
)
from showcase.utils.CentralArgs import CentralArgs
from showcase.utils.constants import feature_set_one
from showcase.utils.helpers import SparkHelpers
from datetime import datetime as dt
# from pyspark.dbutils import DBUtils

def DBUtils(spark):
    pass

def app():

    task_config = {
        "HousePricePrediction": {
            "DataPreProcessing": DataPreProcessingHousePricePredictor,
            "Model": HousePricePredictor,
        }
    }

    # Initialize variables with default values
    ExperimentName = "HousePricePrediction"
    process_name = "Model"
    environment = "dev"
    feature_list = feature_set_one
    process_date = dt.strftime(dt.now(), "%Y-%m-%d")  # Default to current date
    ModelName = "LinearRegression"

    spark = SparkHelpers.get_spark_session("entry-point-init")

    # Retrieve values from Databricks workflows
    widget_names = [
        "ExperimentName",
        "process_name",
        "environment",
        "process_date",
        "ModelName",
        "feature_list",
    ]

    for widget_name in widget_names:
        try:
            if widget_name == feature_list:
                globals()[widget_name] = globals()[
                    DBUtils(spark).widgets.get(widget_name)
                ]
                print("Feature list: ", globals()[widget_name])
            else:
                globals()[widget_name] = DBUtils(spark).widgets.get(widget_name)
        except Exception as e:
            print(f"Failed to retrieve '{widget_name}' widget value: {str(e)}")

    spark = SparkHelpers.get_spark_session(ExperimentName)

    task_config[ExperimentName][process_name](
        CentralArgs(
            spark=spark,
            process_date=process_date,
            ModelName=ModelName,
            FeatureList=feature_list,
            environment = environment
        )
    )

if __name__ == "__main__":
    app()

