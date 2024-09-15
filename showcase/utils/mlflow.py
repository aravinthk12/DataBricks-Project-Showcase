import mlflow
from typing import Any, Optional
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException


def get_mlflow_experiment(
    experiment_id: Optional[str] = None, experiment_name: Optional[str] = None
) -> Experiment:
    """
    Retrieve an MLflow experiment by ID or name.

    Args:
        experiment_id: The ID of the experiment (optional).
        experiment_name: The name of the experiment (optional).

    Returns:
        The MLflow experiment object.

    Raises:
        ValueError: If neither 'experiment_id' nor 'experiment_name' is provided.
    """
    if experiment_id:
        return mlflow.get_experiment(experiment_id)
    if experiment_name:
        return mlflow.get_experiment_by_name(experiment_name)
    raise ValueError("Either 'experiment_id' or 'experiment_name' must be provided")


def mlflow_create_experiment(
    experiment_name: str, artifact_location: str, tags: dict[str, Any]
) -> Experiment:
    """
    Create a new MLflow experiment with the given name and artifact location.

    Args:
        experiment_name: The name of the experiment.
        artifact_location: The location to store experiment artifacts.
        tags: A dictionary of experiment tags.

    Returns:
        The MLflow experiment object.
    """
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_location, tags=tags
        )
    except MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    return get_mlflow_experiment(experiment_id=experiment_id)


def mlflow_delete_experiment(experiment_id: str):
    """
    Delete an MLflow experiment by ID.

    Args:
        experiment_id: The ID of the experiment to delete.
    """
    mlflow.delete_experiment(experiment_id)
