# components/model_evaluation.py
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        # Set MLflow tracking and registry URIs
        os.environ["MLFLOW_TRACKING_URI"] = self.config.mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = "ravteja98"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "62979bfd64ee76fb332fe2e1a2125831e46e4913"  # Replace with new token
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        logger.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

        # Set experiment
        mlflow.set_experiment("test_experiment")

        # Load data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        with mlflow.start_run():
            # Predict and evaluate
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)

            # Save metrics locally
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            logger.info(f"Metrics saved at: {self.config.metric_file_name}")

            # Log metrics and params to MLflow
            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Check tracking URI scheme
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            logger.info(f"Tracking URL type: {tracking_url_type_store}")

            # Log and register model
            if tracking_url_type_store != "file":
                try:
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
                    logger.info("Model registered successfully")
                except Exception as e:
                    logger.error(f"Error registering model: {e}")
                    mlflow.sklearn.log_model(model, "model")  # Fallback to local
            else:
                logger.warning("File-based store detected, logging model locally")
                mlflow.sklearn.log_model(model, "model")