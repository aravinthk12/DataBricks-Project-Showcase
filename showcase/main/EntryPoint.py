from showcase.HousePricePrediction.data_preprocessing.DataPreProcessingHousePricePredictor import (
    DataPreProcessingHousePricePredictor,
)
from showcase.HousePricePrediction.model.HousePricePredictor import (
    HousePricePredictor,
)
from showcase.utils.CentralArgs import CentralArgs
from showcase.utils.helpers import SparkHelpers
from datetime import datetime as dt


if __name__ == "__main__":

    ExperimentName = "HousePricePrediction"

    # Models used
    # 1. "LinearRegression",
    # 2. "DecisionTreeRegressor",
    # 3. "RandomForestRegressor",
    # 4. "GBTRegressor",

    spark = SparkHelpers.get_spark_session(ExperimentName)
    process_date = dt.now().strftime("%Y-%m-%d")

    DataPreProcessingHousePricePredictor(
        CentralArgs(spark=spark, process_date=process_date)
    )
    for ModelName in [
        "LinearRegression",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
        "GBTRegressor",
    ]:
        # ModelName = "LinearRegression"

        HousePricePredictor(
            CentralArgs(
                spark=spark,
                process_date=process_date,
                ModelName=ModelName,
                FeatureList=[
                    "OverallQual",
                    "GrLivArea",
                    "GarageCars",
                    "GarageArea",
                    "TotalBsmtSF",
                    "YearBuilt",
                    "YearRemod/Add",
                    "TotalSF",
                    "PricePerSF",
                    "Qual_LivArea",
                    "GrLivAreaPoly",
                    "LotArea",
                    "FullBath",
                    "Age",
                    "LotFrontage",
                    "Fireplaces",
                    "Neighborhood_Index",
                    "HouseStyle_Index",
                ],
            )
        )
