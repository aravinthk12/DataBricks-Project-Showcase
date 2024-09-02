from showcase.utils.CentralArgs import CentralArgs
from showcase.utils.helpers import read_datasets, write_datasets
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler, PolynomialExpansion, StringIndexer


feature_list_for_vector = [
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
]


class HousePricePredictor:

    def __init__(self, args: CentralArgs):

        self.spark = args.spark
        self.process_date = args.process_date
        self._read_data()
        self._process_date()
        # self._write_data()

    def _read_data(self):
        self.data_dict = read_datasets(
            self.spark, "showcase/HousePricePrediction/features/read.json"
        )

    def _process_date(self):
        self._get_area_feature()
        self._prep_polynomial_features()
        self._encode_cat_features()
        self._handle_missing_values()
        self._prep_feature_vectors()

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

        for col_ in ["LotFrontage", "FullBath"]:
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
            inputCols=feature_list_for_vector, outputCol="features"
        )
        self.data_dict["final_features"] = vector_assembler.transform(
            self.data_dict["price_predict_features_poly"]
        )
