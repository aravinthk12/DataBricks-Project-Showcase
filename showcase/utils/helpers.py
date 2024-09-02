from pyspark.sql import SparkSession, DataFrame
import json
import platform
from typing import Dict, List


class SparkHelpers:
    """
    A utility class for Spark-related helper functions.

    Methods:
        get_spark_session(appName: str = "default") -> SparkSession:
            Creates and returns a SparkSession with the specified application name.
    """

    @staticmethod
    def get_spark_session(appName: str = "default") -> SparkSession:
        """
        Creates and returns a SparkSession with the given application name.

        Args:
            appName (str): The name of the Spark application. Defaults to "default".

        Returns:
            SparkSession: The SparkSession instance.
        """
        return SparkSession.builder.appName(appName).getOrCreate()


def read_json(json_path: str) -> List[Dict]:
    """
    Reads a JSON file and returns its content as a list of dictionaries.

    Args:
        json_path (str): The path to the JSON file to read.

    Returns:
        list: A list of dictionaries representing the JSON data.
    """
    if platform.system() == "Windows":
        additional_path = ""
    else:
        # to provide Databricks path
        additional_path = ""

    with open(f"{additional_path}{json_path}", "r") as files:
        data_list = json.load(files)

    return data_list


def read_datasets(spark: SparkSession, json_path: str) -> Dict[str, DataFrame]:
    """
    Reads datasets specified in a JSON configuration file and returns a dictionary of DataFrames.

    Args:
        spark (SparkSession): The SparkSession instance used to read the data.
        json_path (str): The path to the JSON configuration file that specifies the datasets.

    Returns:
        dict: A dictionary where keys are dataset names and values are Spark DataFrames.

    Raises:
        ValueError: If the 'type' field in the JSON configuration is not "path".
    """
    data_list = read_json(json_path)
    data_dict = {}
    for in_dict in data_list:
        if in_dict["type"] == "path":
            data_dict[in_dict["variable_name"]] = (
                spark.read.format(in_dict["format"])
                .options(**in_dict["options"])
                .load(in_dict["path"])
            )
        else:
            raise ValueError("Please provide a valid 'type'")

    return data_dict


def write_datasets(data_dict: dict, json_path: str) -> None:
    """
    Writes datasets stored in a dictionary to the specified paths and formats as defined in a JSON configuration file.

    This function takes a dictionary of Spark DataFrames and writes each DataFrame to the location and format specified
    in a JSON file. The JSON file should contain a list of dictionaries, each specifying the variable name, format,
    mode (write type), and path for each dataset.

    Args:
        data_dict (dict): A dictionary where the keys are variable names (str) and the values are Spark DataFrames.
        json_path (str): The file path to a JSON configuration file that specifies how and where to save each DataFrame.

    Returns:
        None
    """

    data_list = read_json(json_path)
    for out_dict in data_list:
        (
            data_dict[out_dict["variable_name"]]
            .write.format(out_dict["format"])
            .mode(out_dict["type"])
            .save(out_dict["path"] + "/" + out_dict["variable_name"])
        )

        print(
            f"table name: {out_dict['variable_name']} \n"
            f"write_location: {out_dict['path']}/{out_dict['variable_name']} \n"
            f"status: Success"
        )
