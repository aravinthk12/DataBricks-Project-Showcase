from pyspark.sql import SparkSession, DataFrame
import json
import platform
from typing import Dict, List
import importlib.resources as resources


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


def read_json(json_path: str, file_name: str) -> List[Dict]:
    """
    Reads a JSON file and returns its content as a list of dictionaries.

    The file path is determined based on the operating system. On Windows, it uses a fixed path,
    while on other systems, it uses a dynamic path based on Python's version.

    Args:
        json_path (str): The path to the JSON file to read.
        file_name (str):

    Returns:
        list: A list of dictionaries representing the JSON data.
    """
    if platform.system() == "Windows":
        print("Windows sys")
        resource_ref = json_path + "/" + file_name + ".json"
        with open(f"{resource_ref}", "r") as files:
            data = json.load(files)
    else:
        # to provide Databricks path
        resource_ref = resources.files(f"{json_path.replace('/', '.')}") / f"{file_name}.json"
        with resources.as_file(resource_ref) as file_path:
            # Check if the file exists and load its content
            if file_path.is_file():
                with open(file_path, 'r') as f:
                    data = json.load(f)
            else:
                print(f"File does not exist: {file_path}")

    return data


def read_datasets(spark: SparkSession, env: str, json_path: str, file_name: str) -> Dict[str, DataFrame]:
    """
    Reads datasets specified in a JSON configuration file and returns a dictionary of DataFrames.

    Args:
        spark (SparkSession): The SparkSession instance used to read the data.
        env (str)
        json_path (str): The path to the JSON configuration file that specifies the datasets.
        file_name (Str): read/write

    Returns:
        dict: A dictionary where keys are dataset names and values are Spark DataFrames.

    Raises:
        ValueError: If the 'type' field in the JSON configuration is not "path".
    """
    data_list = read_json(json_path, file_name)
    data_dict = {}
    for in_dict in data_list:

        if platform.system() == "Windows":
            path = in_dict["path"]
        else:
            path = "dbfs:/tables/" + env + '/' + in_dict["path"]

        if in_dict["type"] == "path":
            data_dict[in_dict["variable_name"]] = (
                spark.read.format(in_dict["format"])
                .options(**in_dict["options"])
                .load(path)
            )
        else:
            raise ValueError("Please provide a valid 'type'")

    return data_dict


def write_datasets(data_dict: dict, env: str, json_path: str, file_name: str) -> None:
    """
    Writes datasets stored in a dictionary to the specified paths and formats as defined in a JSON configuration file.

    This function takes a dictionary of Spark DataFrames and writes each DataFrame to the location and format specified
    in a JSON file. The JSON file should contain a list of dictionaries, each specifying the variable name, format,
    mode (write type), and path for each dataset.

    Args:
        data_dict (dict): A dictionary where the keys are variable names (str) and the values are Spark DataFrames.
        env (str): env
        json_path (str): The file path to a JSON configuration file that specifies how and where to save each DataFrame.
        file_name (Str): read/write

    Returns:
        None
    """

    data_list = read_json(json_path, file_name)
    for out_dict in data_list:

        if platform.system() == "Windows":
            path = out_dict["path"]
        else:
            path = "dbfs:/tables/" + env + '/' + out_dict["path"]
        (
            data_dict[out_dict["variable_name"]]
            .write.format(out_dict["format"])
            .mode(out_dict["type"])
            .save(path + "/" + out_dict["variable_name"])
        )

        print(
            f"table name: {out_dict['variable_name']} \n"
            f"write_location: {out_dict['path']}/{out_dict['variable_name']} \n"
            f"status: Success"
        )
