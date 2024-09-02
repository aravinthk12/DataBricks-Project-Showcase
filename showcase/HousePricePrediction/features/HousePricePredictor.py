from showcase.utils.CentralArgs import CentralArgs
from showcase.utils.helpers import read_datasets, write_datasets
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, PolynomialExpansion, StringIndexer
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor,
)
import mlflow
from pyspark.ml.evaluation import RegressionEvaluator
import mlflow.spark


model_config = {
    "LinearRegression": LinearRegression,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "GBTRegressor": GBTRegressor,
}


class HousePricePredictor:

    def __init__(self, args: CentralArgs):

        self.args = args
        self._read_data()
        self._process_data()
        # self._write_data()

    def _read_data(self):
        self.data_dict = read_datasets(
            self.args.spark, "showcase/HousePricePrediction/features/read.json"
        )

    def _process_data(self):

        # Build features
        self._get_area_feature()
        self._prep_polynomial_features()
        self._encode_cat_features()
        self._handle_missing_values()
        self._prep_feature_vectors()

        # Model
        self._run_model()

    def _write_data(self):
        write_datasets(
            self.data_dict,
            "showcase/HousePricePrediction/data_preprocessing/write.json",
        )
        print("overwrite complete")

    def _get_area_feature(self):
        self.data_dict["price_predict_features"] = (
            self.data_dict["ames_housing_pre_processed"]
            # Total house square feet
            .withColumn(
                "TotalSF",
                F.col("GrLivArea") + F.col("TotalBsmtSF") + F.col("GarageArea"),
            )
            # Price per square feet
            .withColumn("PricePerSF", F.col("SalePrice") / F.col("TotalSF"))
            # Interaction terms
            .withColumn("Qual_LivArea", F.col("OverallQual") + F.col("GrLivArea"))
        )

    def _prep_polynomial_features(self):

        vector_assembler = VectorAssembler(
            inputCols=["GrLivArea"], outputCol="GrLivAreaVec"
        )

        self.data_dict["price_predict_features_vec"] = vector_assembler.transform(
            self.data_dict["price_predict_features"]
        )

        polynomial_expansion = PolynomialExpansion(
            degree=2, inputCol="GrLivAreaVec", outputCol="GrLivAreaPoly"
        )

        self.data_dict["price_predict_features_poly"] = polynomial_expansion.transform(
            self.data_dict["price_predict_features_vec"]
        )

    def _encode_cat_features(self):

        indexers = [
            StringIndexer(inputCol=col, outputCol=col + "_Index").fit(
                self.data_dict["price_predict_features_poly"]
            )
            for col in ["Neighborhood", "HouseStyle"]
        ]

        for indexer in indexers:
            self.data_dict["price_predict_features_poly"] = indexer.transform(
                self.data_dict["price_predict_features_poly"]
            )

    def _handle_missing_values(self):

        for col_ in ["LotFrontage", "FullBath", "TotalSF", "PricePerSF"]:
            self.data_dict["price_predict_features_poly"] = self.data_dict[
                "price_predict_features_poly"
            ].fillna(
                {
                    col_: self.data_dict["price_predict_features_poly"].approxQuantile(
                        col_, [0.5], 0.0
                    )[0]
                }
            )

        for col_ in ["Neighborhood_Index", "HouseStyle_Index"]:
            self.data_dict["price_predict_features_poly"] = self.data_dict[
                "price_predict_features_poly"
            ].fillna(
                {
                    col_: self.data_dict["price_predict_features_poly"]
                    .groupBy()
                    .max(col_)
                    .collect()[0][0]
                }
            )

    def _prep_feature_vectors(self):

        vector_assembler = VectorAssembler(
            inputCols=self.args.FeatureList, outputCol="features"
        )
        self.data_dict["final_features"] = vector_assembler.transform(
            self.data_dict["price_predict_features_poly"]
        )

    def _run_model(self):

        self.data_dict["train_data"], self.data_dict["test_data"] = self.data_dict[
            "final_features"
        ].randomSplit([0.8, 0.2], seed=42)

        target = "SalePrice"

        with mlflow.start_run(run_name=self.args.ModelName):

            # Train the model
            model_fit = model_config[self.args.ModelName](
                labelCol=target, featuresCol="features"
            ).fit(self.data_dict["train_data"])

            # Prediction
            predictions = model_fit.transform(self.data_dict["test_data"])

            # Evaluate the Model
            evaluator = RegressionEvaluator(
                labelCol=target, predictionCol="prediction", metricName="rmse"
            )
            rmse = evaluator.evaluate(predictions)

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.spark.log_model(model_fit, "model")
