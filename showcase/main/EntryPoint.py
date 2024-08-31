from showcase.HousePricePrediction.data_preprocessing.DataPreProcessingHousePricePredictor import (
    DataPreProcessingHousePricePredictor,
)
from showcase.utils.CentralArgs import CentralArgs
from showcase.utils.helpers import SparkHelpers
from datetime import datetime as dt


if __name__ == "__main__":
    spark = SparkHelpers.get_spark_session("HousePricePrediction")
    process_date = dt.now().strftime("%Y-%m-%d")

    DataPreProcessingHousePricePredictor(
        CentralArgs(spark=spark, process_date=process_date)
    )
