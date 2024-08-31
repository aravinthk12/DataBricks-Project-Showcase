from showcase.utils.CentralArgs import CentralArgs
from showcase.utils.helpers import read_datasets, write_datasets
import pyspark.sql.functions as F


class DataPreProcessingHousePricePredictor:
    """
    A class to handle the preprocessing of house price prediction data using Apache Spark.

    Attributes:
        spark (SparkSession): The Spark session instance used for data processing.
        process_date (str): The date used for processing, formatted as 'YYYY-MM-DD'.
        data_dict (dict): A dictionary containing DataFrames loaded from the input data.

    Methods:
        _load_data(): Loads data into the `data_dict` attribute from a specified JSON source.
        _data_preprocessing(): Performs data preprocessing, such as calculating the age of properties.

    Parameters:
        args (CentralArgs): An object containing Spark session and process date.
    """

    def __init__(self, args: CentralArgs):
        """
        Initializes the DataPreProcessingHousePricePredictor with the provided arguments.

        Args:
            args (CentralArgs): An object with the following attributes:
                - spark (SparkSession): The Spark session instance.
                - process_date (str): The date used for processing, formatted as 'YYYY-MM-DD'.
        """
        self.spark = args.spark
        self.process_date = args.process_date
        self._read_data()
        self._process_date()
        self._write_data()

    def _read_data(self):
        """
        Loads data into the `data_dict` attribute from the JSON file located at
        "showcase/HousePricePrediction/data_preprocessing/read.json".
        """
        self.data_dict = read_datasets(
            self.spark, "showcase/HousePricePrediction/data_preprocessing/read.json"
        )

    def _process_date(self):
        """
        Performs data preprocessing by calculating the age of properties. Adds a new column 'Age' to the
        DataFrame in `data_dict` under the key 'ames_housing_raw'. The age is calculated as the difference
        between the `process_date` year and the 'Year Built' column.
        Additionally, this method removes any spaces from the column names in the DataFrame to ensure consistent naming.
        After processing, the modified DataFrame is stored in `data_dict` under the key 'ames_housing_pre_processed'.

        """
        self.data_dict["ames_housing_pre_processed"] = (
            self.data_dict["ames_housing_raw"]
            .withColumn("Age", self.process_date[:4] - F.col("Year Built")))

        for col_ in self.data_dict["ames_housing_pre_processed"].columns:
            self.data_dict["ames_housing_pre_processed"] = (
                self.data_dict["ames_housing_pre_processed"]
                .withColumn(col_.replace(' ', ''), F.col(col_)))

    def _write_data(self):
        """
            Writes the preprocessed datasets to their respective locations as defined in the JSON configuration file.

            This method uses the `write_datasets` function to write all datasets stored in `self.data_dict` to the locations
            and formats specified in the JSON file located at "showcase/HousePricePrediction/data_preprocessing/write.json".

            After successfully writing the data, it prints a confirmation message indicating that the overwrite operation
            is complete.
        """

        write_datasets(self.data_dict, "showcase/HousePricePrediction/data_preprocessing/write.json")
        print("overwrite complete")



