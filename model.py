import os
import warnings
import json
from io import BytesIO
from dotenv import load_dotenv; load_dotenv("config.env")
import boto3
from botocore.auth import NoCredentialsError
import pandas as pd
import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
# from mlflow.models.signature import infer_signature
from autogluon.tabular import TabularPredictor
from mlflow import MlflowClient
import mlflow
from mlflow.models import infer_signature
from PIL import Image
import shap
import matplotlib.pyplot as plt

       
class AutogluonModel(mlflow.pyfunc.PythonModel):
    """
    MLflow Python model for Autogluon.

    This class is used to load and predict using an Autogluon model.

    Attributes:
        predictor (TabularPredictor): The Autogluon TabularPredictor object.

    Methods:
        load_context(self, context): Load the Autogluon model from MLflow artifacts.
        predict(self, context, model_input): Make predictions using the loaded model.
    """
    def load_context(self, context):
        """
        Load the Autogluon model from MLflow artifacts.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The context object containing model artifacts.

        Returns:
            None
        """
        self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))

    def predict(self, context, model_input):
        """
        Make predictions using the loaded Autogluon model.

        Args:
            context (mlflow.pyfunc.PythonModelContext): The context object containing model artifacts.
            model_input (pandas.DataFrame): Input data for making predictions.

        Returns:
            numpy.ndarray: Predicted values.
        """
        return self.predictor.predict(model_input, as_pandas=False)


def plot_to_numpy():
    """
    Convert the current matplotlib plot into a numpy array, as then we can
    pass it to mlflow.log_image() easily.
    """
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = np.array(Image.open(buf))
    buf.close()

    return img


def convert_ndarray_to_pd_dataframe(ndarray, column_names):
    return pd.DataFrame(data=ndarray, columns=column_names)


def train(
    experiment_name=None,
    model_name='my_model', 
    data_path='data.csv',
    target_column='label', 
    problem_type=None,
    hyperparameters = {'XGB': {}},
    time_limit=1 * 60,
    fit_weighted_ensemble=False,
    extra_pip_requirements=None
):
    """
    Train and register an Autogluon model using MLflow.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        model_name (str): Name for the registered model.
        data_path (str): Path to the CSV file containing the training data.
        target_column (str): Name of the target column in the dataset.
        problem_type (str): Type of machine learning problem (e.g., 'classification', 'regression').
        hyperparameters (dict): Hyperparameters for the Autogluon model.
        time_limit (int): Time limit for model training in seconds.
        fit_weighted_ensemble (bool): Whether to fit a weighted ensemble.
        extra_pip_requirements (list): Additional pip requirements for the model's environment.

    Returns:
        TabularPredictor: Trained Autogluon model.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URL"))
    if not experiment_name:
        experiment_name = 'Default'
    experiment = mlflow.set_experiment(experiment_name)
    object_storage = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
        config=boto3.session.Config(signature_version="s3v4"),
    )
    data = pd.read_csv(data_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train_data, test_data = train_test_split(data, test_size=.2, random_state=42)
    
    # train and save model
    with mlflow.start_run():
        predictor = (
            TabularPredictor(label = target_column, problem_type=problem_type)
            .fit(
                train_data,
                time_limit=time_limit,
                hyperparameters=hyperparameters,
                fit_weighted_ensemble=fit_weighted_ensemble,
            )
        )


        trainer = predictor._learner.load_trainer()
        autogluon_model_name = trainer._get_best()
        
        model_autogluon = trainer.load_model(model_name=autogluon_model_name)
        raw_model = model_autogluon.model

        train_data_no_labels = train_data.drop(train_data.columns[-1], axis=1)

        subset_size = 100
        if autogluon_model_name in ['RandomForest', 'ExtraTrees', 'LightGBM', 'CatBoost', 'XGBoost']:
            explainer = shap.TreeExplainer(raw_model, train_data_no_labels[:subset_size])
        elif autogluon_model_name in ['LinearModel']:
            explainer = shap.LinearExplainer(raw_model, train_data_no_labels[:subset_size])
        elif autogluon_model_name in ['NeuralNetTorch']:
            # TODO
            explainer = shap.DeepExplainer(raw_model, train_data_no_labels[:subset_size])
        else:
            explainer = shap.KernelExplainer(raw_model.predict, train_data_no_labels[:subset_size])    
        
        shap_values = explainer(train_data_no_labels, check_additivity=False) 
        
        # predictions = predictor.predict(test_data)
        # signature = infer_signature(test_data.drop(columns=[target_column], axis=1), predictions)
        metrics = predictor.evaluate(test_data, silent=True)
        mlflow.log_metrics(metrics)

        if problem_type in ['binary', 'regression']:
            shap.plots.beeswarm(shap_values, show=False)
            beeswarm = plot_to_numpy()
            plt.clf()
            shap.plots.waterfall(shap_values[0], show=False)
            waterfall = plot_to_numpy()
        else:
            shap.plots.beeswarm(shap_values[:, :, 0], show=False)
            beeswarm = plot_to_numpy()
            plt.clf()
            shap.plots.waterfall(shap_values[0, :, 0], show=False)
            waterfall = plot_to_numpy()

        mlflow.log_image(beeswarm, 'beeswarm.png')
        mlflow.log_image(waterfall, 'waterfall.png')
        
        artifacts = {"predictor_path": predictor.path}
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=AutogluonModel(),
            artifacts=artifacts,
            registered_model_name=model_name,
            # signature=signature,
            extra_pip_requirements=extra_pip_requirements,
            code_path=["shared/src/"]
        )
    client = MlflowClient(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URL")
    )
    models = client.get_latest_versions(model_name)
    model = max(models, key=lambda x: int(x.version))
    model_settings = {
        "name": model_name,
        "implementation": "mlserver_mlflow.MLflowRuntime",
        "parameters": {
            "uri": "."
        }
    }
    json_str = json.dumps(model_settings)
    bytes_data = json_str.encode('utf-8')
    file_obj = BytesIO(bytes_data)
    response = object_storage.upload_fileobj(file_obj, "mlflow", f"{model.source.split(':')[1]}/model-settings.json")
    return predictor
