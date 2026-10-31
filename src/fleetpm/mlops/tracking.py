
import hashlib
import inspect
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ..config import art


def setup(cfg):
    """Configure MLflow to use the project's local tracking database and artifacts."""

    tracking_db = art(cfg, "mlflow.db").resolve()
    artifact_dir = art(cfg, "mlartifacts").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Configure the local SQLite tracking database.
    mlflow.set_tracking_uri(
        f"sqlite:///{tracking_db.as_posix()}"
    )

    # The experiment name is keyed to this machine's artifact directory, so a tracking database
    # copied from another machine (for example file:///C:/... locations) can never supply a foreign path.
    location = artifact_dir.as_uri()
    name = f"{cfg['mlops']['experiment']}_{hashlib.sha1(location.encode()).hexdigest()[:6]}"

    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        mlflow.create_experiment(name, artifact_location=location)
    elif experiment.artifact_location != location:
        raise RuntimeError(
            f"Experiment {name} has artifact location {experiment.artifact_location}, expected {location}"
        )

    mlflow.set_experiment(name)


XGB_TRUSTED = [
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBRegressor",
]


def log_sk_model(model):
    """Log a scikit-learn-compatible model and return its MLflow model URI."""

    # MLflow 2.x uses artifact_path; newer versions may use name.
    kw = (
        "name"
        if "name" in inspect.signature(
            mlflow.sklearn.log_model
        ).parameters
        else "artifact_path"
    )

    return mlflow.sklearn.log_model(
        model,
        **{kw: "model"},
        skops_trusted_types=XGB_TRUSTED,
    ).model_uri


def register_and_maybe_promote(
    model_uri,
    run_id,
    reg_name,
    metric,
    margin=0.0,
):
    """Register a model and promote it if its metric beats the champion."""

    client = MlflowClient()

    model_version = mlflow.register_model(
        model_uri,
        reg_name,
    )

    new_score = client.get_run(run_id).data.metrics[metric]

    try:
        champion = client.get_model_version_by_alias(
            reg_name,
            "champion",
        )
        old_score = client.get_run(
            champion.run_id
        ).data.metrics[metric]

    except MlflowException:
        old_score = None

    if old_score is None or new_score < old_score * (1 - margin):
        client.set_registered_model_alias(
            reg_name,
            "champion",
            model_version.version,
        )
        return True, model_version.version

    client.set_registered_model_alias(
        reg_name,
        "challenger",
        model_version.version,
    )
    return False, model_version.version
