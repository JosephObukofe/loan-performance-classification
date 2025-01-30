import os
import io
import boto3
import pickle
import logging
import tempfile
import mlflow.tracking
import numpy as np
import pandas as pd
import urllib.parse
import mlflow
import mlflow.sklearn
import mlflow.tracking
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from botocore.exceptions import BotoCoreError, ClientError
from minio import Minio
from minio.error import MinioException, S3Error
from datetime import datetime
from typing import List, Dict, Any, Tuple, Union, Optional, Callable
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    roc_auc_score,
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from user_performance_classification.config.config import (
    MINIO_URL,
    MINIO_ROOT_USER,
    MINIO_ROOT_PASSWORD,
    MLFLOW_TRACKING_URI,
)


class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_type="box-cox"):
        """
        Custom transformer for applying numerical feature transformations.

        Parameters
        ----------
        encoding_type : str, default="box-cox"
            The type of transformation to apply. Can be either "box-cox", "yeo-johnson", "sqrt", or "none".
        """

        self.encoding_type = encoding_type
        self.pt = None  # For PowerTransformer

    def fit(self, X, y=None):
        """
        Fits the transformer to the data.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            The input data to fit.
        y : Ignored
            Not used, present here for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """

        if self.encoding_type == "box-cox":
            if (X <= 0).any().any():
                raise ValueError(
                    "Box-Cox transformation requires all data to be positive."
                )
        elif self.encoding_type == "yeo-johnson":
            # Initialize PowerTransformer for Yeo-Johnson
            self.pt = PowerTransformer(method="yeo-johnson")
            self.pt.fit(X)
        return self

    def transform(self, X):
        """
        Applies the transformation to the data.

        Parameters
        ----------
        X : array-like or pandas DataFrame
            The input data to transform.

        Returns
        -------
        X_transformed : array-like
            Transformed data.
        """

        if self.encoding_type == "box-cox":
            return np.log(X) if (X > 0).all().all() else np.log1p(X)
        elif self.encoding_type == "yeo-johnson":
            return self.pt.transform(X)
        elif self.encoding_type == "sqrt":
            return np.sqrt(X)
        elif self.encoding_type == "none":
            return X
        else:
            raise ValueError(f"Unknown encoding_type: {self.encoding_type}")

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """

        return {"encoding_type": self.encoding_type}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Returns
        -------
        self : object
            Returns self.
        """

        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError(
                "input_features must be provided to generate feature names."
            )
        return [f"{col}_transformed" for col in input_features]


def sklearn_tags_patch(self):
    """
    Returns a dictionary of tags used to indicate certain properties of a scikit-learn estimator.

    Returns
    -------
    dict: A dictionary containing the following tags:
        - "non_deterministic" (bool): Indicates if the estimator is non-deterministic.
        - "binary_only" (bool): Indicates if the estimator only supports binary classification.
        - "requires_positive_X" (bool): Indicates if the estimator requires positive input features.
        - "requires_positive_y" (bool): Indicates if the estimator requires positive target values.
        - "X_types" (list): Specifies the types of input features supported by the estimator.
        - "poor_score" (bool): Indicates if the estimator is expected to have poor performance.
    """

    return {
        "non_deterministic": True,
        "binary_only": False,
        "requires_positive_X": False,
        "requires_positive_y": False,
        "X_types": ["2darray"],
        "poor_score": False,
    }


class SklearnCompatibleXGBClassifier(XGBClassifier):
    """
    A custom classifier that extends the XGBClassifier to be compatible with scikit-learn's interface.

    This class overrides the `__sklearn_tags__` method to ensure compatibility with scikit-learn's estimator tags, which are used for various checks and functionalities within the scikit-learn ecosystem.

    Methods
    -------
    __sklearn_tags__():
        Returns the scikit-learn compatible tags for this estimator.
    """

    def __sklearn_tags__(self):
        return sklearn_tags_patch()


def preprocess_training_data(
    X: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
    encoding_type: str = "box-cox",
) -> Tuple[pd.DataFrame, Pipeline, List[str]]:
    """
    Preprocesses the input DataFrame by:
    - Applying a selected transformation (Box-Cox, Yeo-Johnson, square root, or none) to numeric features.
    - Applying one-hot encoding to categorical features.
    - Scaling all transformed features using StandardScaler.

    Parameters
    ----------
    X : pd.DataFrame
        The input DataFrame containing features to preprocess.
    numeric_features : list
        List of numerical features to apply the transformation.
    categorical_features : list
        List of categorical features to one-hot encode.
    encoding_type : str, default="box-cox"
        The type of transformation to apply. Can be "box-cox", "yeo-johnson", "sqrt", or "none".

    Returns
    -------
    Tuple[pd.DataFrame, Pipeline, List[str]]
        - pd.DataFrame: Preprocessed DataFrame with transformed, encoded, and scaled features.
        - Pipeline: Fitted pipeline for transforming the test set.
        - List[str]: List of column names for the transformed dataset.
    """

    transformer = Transformer(encoding_type=encoding_type)
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric_transform", transformer, numeric_features),
            ("onehot_encode", one_hot_encoder, categorical_features),
        ]
    )

    # Complete pipeline with scaling
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("scaling", StandardScaler()),
        ]
    )

    processed_array = pipeline.fit_transform(X)

    if encoding_type == "none":
        numeric_column_names = numeric_features
    else:
        numeric_column_names = [
            f"{feature}_{encoding_type}" for feature in numeric_features
        ]

    onehot_encoder = pipeline.named_steps["preprocessing"].named_transformers_[
        "onehot_encode"
    ]

    onehot_column_names = [
        f"{feature}_{category}"
        for feature, categories in zip(categorical_features, onehot_encoder.categories_)
        for category in categories[1:]  # Exclude first category (dropped)
    ]

    processed_column_names = numeric_column_names + onehot_column_names

    if processed_array.shape[1] != len(processed_column_names):
        raise ValueError(
            f"Shape mismatch: processed data has {processed_array.shape[1]} columns, "
            f"but {len(processed_column_names)} column names were generated."
        )

    processed_df = pd.DataFrame(processed_array, columns=processed_column_names)
    return processed_df, pipeline, processed_column_names


def preprocess_test_data(
    X: pd.DataFrame,
    pipeline: Pipeline,
    processed_column_names: List[str],
) -> pd.DataFrame:
    """
    Preprocesses the test DataFrame using a fitted pipeline with specified transformations.

    This function applies transformations learned during training to the test data without fitting on it,
    ensuring no data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        The input test DataFrame containing features to preprocess.
    pipeline : Pipeline
        The fitted preprocessing pipeline from the training data.
    processed_column_names : List[str]
        List of column names generated during the training preprocessing.

    Returns
    -------
    pd.DataFrame
        A preprocessed test DataFrame with transformed, encoded, and scaled features.
    """

    processed_array = pipeline.transform(X)

    if processed_array.shape[1] != len(processed_column_names):
        raise ValueError(
            f"Shape mismatch: processed data has {processed_array.shape[1]} columns, "
            f"but {len(processed_column_names)} column names were expected."
        )

    processed_df = pd.DataFrame(processed_array, columns=processed_column_names)
    return processed_df


def capitalize_and_replace(
    s: str,
    delimiter: str,
) -> str:
    """
    Splits the input string by the given delimiter, capitalizes each word, and joins them with a space.

    Args:
        s (str): The input string to be processed.
        delimiter (str): The delimiter to split the input string.

    Returns
    -------
    str
        A string with each word capitalized and joined by a space.
    """
    words = s.split(delimiter)
    capitalized_words = [word.capitalize() for word in words]
    return " ".join(capitalized_words)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path : str
        The path to the CSV file that contains the data to be loaded.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame containing the data from the specified CSV file.
    """

    raw_data = pd.read_csv(file_path)
    return raw_data


def read_csv_from_minio(
    bucket_name: str,
    object_name: str,
) -> pd.DataFrame:
    """
    Reads a CSV file from MinIO and loads it into a pandas DataFrame.

    Parameters
    ----------
    bucket_name : str
        The name of the MinIO bucket where the CSV file is stored.
    object_name : str
        The name of the CSV file (object) in the MinIO bucket.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the data from the CSV file.

    Raises
    -------
    Exception
        For any errors that may occur while reading the CSV file.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )

        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        df = pd.read_csv(io.BytesIO(response["Body"].read()))

        logger.info(
            f"Successfully loaded CSV data from '{object_name}' in bucket '{bucket_name}'."
        )
        return df
    except Exception as e:
        logger.error(f"Error reading CSV file '{object_name}': {e}")
        return None


def upload_data_to_minio(
    dataframe: pd.DataFrame,
    bucket_name: str,
    object_name: str,
) -> None:
    """
    Uploads a pandas DataFrame to MinIO as a pickled object.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The pandas DataFrame to be uploaded to MinIO.
    bucket_name : str
        The name of the MinIO bucket where the pickled dataframe will be stored.
    object_name : str
        The name of the pickled dataframe (object) in the MinIO bucket.

    Returns
    -------
    None
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )

        with io.BytesIO() as buffer:
            pickle.dump(dataframe, buffer)
            buffer.seek(0)

            s3_client.upload_fileobj(
                Fileobj=buffer,
                Bucket=bucket_name,
                Key=object_name,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )

            logger.info(
                f"Successfully uploaded DataFrame to MinIO bucket '{bucket_name}' as '{object_name}'"
            )
    except Exception as e:
        logger.error(f"An error occurred while uploading data to MinIO: {e}")


def load_data_from_minio(
    bucket_name: str,
    object_name: str,
) -> pd.DataFrame:
    """
    Load a pickled pandas DataFrame from MinIO.

    Parameters
    ----------
    bucket_name : str
        The name of the MinIO bucket where the pickled file is stored.
    object_name : str
        The name of the pickled object (file) in the MinIO bucket.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame deserialized from the pickled file.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        s3_client = boto3.client(
            "s3",
            endpoint_url=MINIO_URL,
            aws_access_key_id=MINIO_ROOT_USER,
            aws_secret_access_key=MINIO_ROOT_PASSWORD,
        )

        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)

        with io.BytesIO(response["Body"].read()) as buffer:
            dataframe = pickle.load(buffer)

        logger.info(
            f"Successfully loaded pickled DataFrame '{object_name}' from bucket '{bucket_name}'."
        )
        return dataframe
    except Exception as e:
        logger.error(f"An error occurred while loading data from MinIO: {e}")
        raise e


def mutual_info(
    df: pd.DataFrame,
    n_bins=10,
) -> pd.DataFrame:
    """
    Calculate the mutual information matrix for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame for which to calculate the mutual information matrix.
    n_bins : int, optional
        The number of bins to use for discretizing the data, by default 10.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the mutual information scores between each pair of columns in the input DataFrame.
    """

    mi_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    df_discrete = pd.DataFrame(discretizer.fit_transform(df), columns=df.columns)

    for col1 in df.columns:
        for col2 in df.columns:
            if col1 != col2:
                mi_score = mutual_info_regression(
                    df_discrete[[col1]], df_discrete[col2]
                )
                mi_matrix.loc[col1, col2] = mi_score[0]
            else:
                mi_matrix.loc[col1, col2] = 0

    return mi_matrix


def custom_scoring(
    y_true: np.ndarray,
    y_pred: np.ndarray = None,
    y_proba: np.ndarray = None,
    needs_proba: bool = False,
) -> float:
    """
    Custom scoring function that computes a weighted score based on precision, F1 score,
    accuracy, and recall.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like, optional
        Estimated target values returned by the classifier.
    y_proba : array-like, optional
        Predicted probabilities returned by the classifier.
    needs_proba : bool, optional
        Whether the scoring function requires probabilities (default is False).

    Returns
    -------
    float
        A weighted score combining precision, F1 score, accuracy, and recall or a ROC AUC score.
    """

    if needs_proba and y_proba is not None:
        return roc_auc_score(y_true, y_proba[:, 1])

    if y_pred is None:
        raise ValueError("y_pred must be provided when 'needs_proba' is False.")

    precision_weight = 0.4
    f1_weight = 0.3
    accuracy_weight = 0.2
    recall_weight = 0.1

    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return (
        (precision_weight * precision)
        + (f1_weight * f1)
        + (accuracy_weight * accuracy)
        + (recall_weight * recall)
    )


def custom_scorer() -> make_scorer:
    """
    Wrapper function to create a custom scorer using the `custom_scoring` function.

    Returns
    -------
    scorer
        A scorer object to be used with GridSearchCV or other model evaluation tools.
        The scorer uses the custom scoring function and ensures that higher scores are better.
    """
    return make_scorer(
        custom_scoring,
        needs_proba=True,
        greater_is_better=True,
    )


def bayesian_hyperparameter_tuning(
    estimator: ClassifierMixin,
    search_spaces: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> ClassifierMixin:
    """
    Performs hyperparameter tuning for a given classification model using BayesSearchCV.

    Parameters
    ----------
    estimator : ClassifierMixin
        The classification model to tune (e.g., LogisticRegression, DecisionTreeClassifier).
    search_spaces : Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.

    Returns
    -------
    ClassifierMixin
        The best classifier model found by BayesSearchCV after hyperparameter tuning.
    """

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=3,
        random_state=42,
    )

    bayes_search = BayesSearchCV(
        estimator=estimator,
        search_spaces=search_spaces,
        scoring=custom_scorer(),
        cv=cv,
        refit=True,
        random_state=42,
        n_iter=50,
        n_jobs=6,
        verbose=1,
    )

    try:
        bayes_search.fit(X_train, y_train)
    except AttributeError as e:
        raise ValueError(f"Error with estimator '{estimator.__class__.__name__}': {e}")

    tuned_model = bayes_search.best_estimator_
    return tuned_model


def train_logistic_regression_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: Dict[str, Any],
    dataset_type: str,
) -> LogisticRegression:
    """
    Trains a Logistic Regression model with Bayesian search for hyperparameter optimization.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    search_space: Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    dataset_type: str
        The training data type the model was trained on. Could be transformed, binned_labeled or binned_encoded

    Returns
    -------
    LogisticRegression
        The best Logistic Regression model found after hyperparameter tuning.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Logistic Regression Classifier Experiment Run w/ Class Weights - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Loan Performance Logistic Regression Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        logistic_regression_model = LogisticRegression(
            class_weight={0: 1, 1: 4},
            solver="saga",
            random_state=42,
        )

        logistic_regression_tuned_model = bayesian_hyperparameter_tuning(
            estimator=logistic_regression_model,
            search_spaces=search_space,
            X_train=X_train,
            y_train=y_train,
        )

        logistic_regression_best_params = logistic_regression_tuned_model.get_params()
        logistic_regression_tuned_model.fit(X_train, y_train)
        logistic_regression_pred = logistic_regression_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, logistic_regression_pred)
        precision = precision_score(y_train, logistic_regression_pred)
        recall = recall_score(y_train, logistic_regression_pred)
        f1 = f1_score(y_train, logistic_regression_pred)

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=logistic_regression_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            logger.error(f"Error logging model tag: {e}")

        try:
            mlflow.log_params(logistic_regression_best_params)
        except Exception as e:
            logger.error(f"Error logging best model parameters: {e}")

        try:
            mlflow.log_param("dataset_type", dataset_type)
        except Exception as e:
            logger.error(f"Error logging dataset type: {e}")

        try:
            mlflow.log_metrics(
                {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                }
            )
        except Exception as e:
            logger.error(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Logistic Regression Classifier for Loan Performance Classification with "
            f"best parameters: {logistic_regression_best_params}. "
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            logger.error(f"Model registration for {model_name} failed: {e}")

    return logistic_regression_tuned_model


def train_decision_tree_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: Dict[str, Any],
    dataset_type: str,
) -> DecisionTreeClassifier:
    """
    Trains a Decision Tree model with Bayesian search for hyperparameter optimization.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    search_space: Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    dataset_type: str
        The training data type the model was trained on. Could be transformed, binned_labeled or binned_encoded

    Returns
    -------
    DecisionTreeClassifier
        The best Decision Tree model found after hyperparameter tuning.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Decision Tree Classifier Experiment Run w/ Class Weights - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Loan Performance Decision Tree Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        decision_tree_model = DecisionTreeClassifier(
            class_weight={0: 1, 1: 4},
            random_state=42,
        )

        decision_tree_tuned_model = bayesian_hyperparameter_tuning(
            estimator=decision_tree_model,
            search_spaces=search_space,
            X_train=X_train,
            y_train=y_train,
        )

        decision_tree_best_params = decision_tree_tuned_model.get_params()
        decision_tree_tuned_model.fit(X_train, y_train)
        decision_tree_pred = decision_tree_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, decision_tree_pred)
        precision = precision_score(y_train, decision_tree_pred)
        recall = recall_score(y_train, decision_tree_pred)
        f1 = f1_score(y_train, decision_tree_pred)

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=decision_tree_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            logger.error(f"Error logging model tag: {e}")

        try:
            mlflow.log_params(decision_tree_best_params)
        except Exception as e:
            logger.error(f"Error logging best model parameters: {e}")

        try:
            mlflow.log_param("dataset_type", dataset_type)
        except Exception as e:
            logger.error(f"Error logging dataset type: {e}")

        try:
            mlflow.log_metrics(
                {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                }
            )
        except Exception as e:
            logger.error(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Decision Tree Classifier for Loan Performance Classification with "
            f"best parameters: {decision_tree_best_params}. "
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            logger.error(f"Model registration for {model_name} failed: {e}")

    return decision_tree_tuned_model


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: Dict[str, Any],
    dataset_type: str,
) -> RandomForestClassifier:
    """
    Trains a Random Forest model with Bayesian search for hyperparameter optimization.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    search_space: Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    dataset_type: str
        The training data type the model was trained on. Could be transformed, binned_labeled or binned_encoded

    Returns
    -------
    RandomForestClassifier
        The best Random Forest model found after hyperparameter tuning.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Random Forest Classifier Experiment Run w/ Class Weights - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Loan Performance Random Forest Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        random_forest_model = RandomForestClassifier(
            class_weight={0: 1, 1: 4},
            bootstrap=True,
            random_state=42,
        )

        random_forest_tuned_model = bayesian_hyperparameter_tuning(
            estimator=random_forest_model,
            search_spaces=search_space,
            X_train=X_train,
            y_train=y_train,
        )

        random_forest_best_params = random_forest_tuned_model.get_params()
        random_forest_tuned_model.fit(X_train, y_train)
        random_forest_pred = random_forest_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, random_forest_pred)
        precision = precision_score(y_train, random_forest_pred)
        recall = recall_score(y_train, random_forest_pred)
        f1 = f1_score(y_train, random_forest_pred)

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=random_forest_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            logger.error(f"Error logging model tag: {e}")

        try:
            mlflow.log_params(random_forest_best_params)
        except Exception as e:
            logger.error(f"Error logging best model parameters: {e}")

        try:
            mlflow.log_param("dataset_type", dataset_type)
        except Exception as e:
            logger.error(f"Error logging dataset type: {e}")

        try:
            mlflow.log_metrics(
                {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                }
            )
        except Exception as e:
            logger.error(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Random Forest Classifier for Loan Performance Classification with "
            f"best parameters: {random_forest_best_params}. "
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            logger.error(f"Model registration for {model_name} failed: {e}")

    return random_forest_tuned_model


def train_svm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: Dict[str, Any],
    dataset_type: str,
) -> SVC:
    """
    Trains a Support Vector Classifier model with Bayesian search for hyperparameter optimization.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    search_space: Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    dataset_type: str
        The training data type the model was trained on. Could be transformed, binned_labeled or binned_encoded

    Returns
    -------
    SVC
        The best Support Vector Classifier model found after hyperparameter tuning.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Support Vector Classifier Experiment Run w/ Class Weights - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Loan Performance Support Vector Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        svc_model = SVC(
            class_weight={0: 1, 1: 4},
            random_state=42,
        )

        svc_tuned_model = bayesian_hyperparameter_tuning(
            estimator=svc_model,
            search_spaces=search_space,
            X_train=X_train,
            y_train=y_train,
        )

        svc_best_params = svc_tuned_model.get_params()
        svc_tuned_model.fit(X_train, y_train)
        svc_pred = svc_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, svc_pred)
        precision = precision_score(y_train, svc_pred)
        recall = recall_score(y_train, svc_pred)
        f1 = f1_score(y_train, svc_pred)

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=svc_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            logger.error(f"Error logging model tag: {e}")

        try:
            mlflow.log_params(svc_best_params)
        except Exception as e:
            logger.error(f"Error logging best model parameters: {e}")

        try:
            mlflow.log_param("dataset_type", dataset_type)
        except Exception as e:
            logger.error(f"Error logging dataset type: {e}")

        try:
            mlflow.log_metrics(
                {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                }
            )
        except Exception as e:
            logger.error(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Support Vector Classifier for Loan Performance Classification with "
            f"best parameters: {svc_best_params}. "
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            logger.error(f"Model registration for {model_name} failed: {e}")

    return svc_tuned_model


def train_hgboost_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    search_space: Dict[str, Any],
    dataset_type: str,
) -> HistGradientBoostingClassifier:
    """
    Trains a Histogram Gradient Boosting Classifier model with Bayesian search for hyperparameter optimization.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data containing feature variables.
    y_train : pd.Series
        Target labels corresponding to X_train.
    search_space: Dict[str, Any]
        A dictionary defining the search space for each hyperparameter.
    dataset_type: str
        The training data type the model was trained on. Could be transformed, binned_labeled or binned_encoded

    Returns
    -------
    HistGradientBoostingClassifier
        The best Histogram Gradient Boosting Classifier model found after hyperparameter tuning.
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = (
        f"Histogram Gradient Boosting Classifier Experiment Run w/ Class Weights - "
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    registered_model_name = "Loan Performance Histogram Gradient Boosting Classifier"

    with mlflow.start_run(run_name=model_name) as run:
        hgboost_model = HistGradientBoostingClassifier(
            class_weight={0: 1, 1: 4},
            early_stopping=True,
            random_state=42,
        )

        hgboost_tuned_model = bayesian_hyperparameter_tuning(
            estimator=hgboost_model,
            search_spaces=search_space,
            X_train=X_train,
            y_train=y_train,
        )

        hgboost_best_params = hgboost_tuned_model.get_params()
        hgboost_tuned_model.fit(X_train, y_train)
        hgboost_pred = hgboost_tuned_model.predict(X_train)
        accuracy = accuracy_score(y_train, hgboost_pred)
        precision = precision_score(y_train, hgboost_pred)
        recall = recall_score(y_train, hgboost_pred)
        f1 = f1_score(y_train, hgboost_pred)

        artifact_path = f"{registered_model_name.lower().replace("(", "").replace(")", "").replace(" ", "_")}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            mlflow.sklearn.log_model(
                sk_model=hgboost_tuned_model,
                artifact_path=artifact_path,
            )
        except mlflow.exceptions.MlflowException as e:
            logger.error(f"Error logging model: {e}")

        try:
            mlflow.set_tag("artifact_path", artifact_path)
        except Exception as e:
            logger.error(f"Error logging model tag: {e}")

        try:
            mlflow.log_params(hgboost_best_params)
        except Exception as e:
            logger.error(f"Error logging best model parameters: {e}")

        try:
            mlflow.log_param("dataset_type", dataset_type)
        except Exception as e:
            logger.error(f"Error logging dataset type: {e}")

        try:
            mlflow.log_metrics(
                {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                }
            )
        except Exception as e:
            logger.error(f"Error logging metric: {e}")

        model_uri = f"runs:/{run.info.run_id}/{urllib.parse.quote(model_name)}"
        model_description = (
            f"Histogram Gradient Boosting Classifier for Loan Performance Classification with "
            f"best parameters: {hgboost_best_params}. "
        )

        try:
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=registered_model_name,
            )

            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=registered_model_name,
                version=registered_model.version,
                description=model_description,
            )

            client.set_model_version_tag(
                name=registered_model_name,
                version=registered_model.version,
                key="Trained at",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            client.set_registered_model_alias(
                name=registered_model_name,
                version=registered_model.version,
                alias="Staging",
            )
        except MlflowException as e:
            logger.error(f"Model registration for {model_name} failed: {e}")

    return hgboost_tuned_model


def load_model_from_mlflow(
    run_id: str,
    artifact_path: str,
):
    """
    Loads a model from MLflow given the run ID and artifact path.

    Parameters
    ----------
    run_id : str
        The run ID of the MLflow run where the model is logged.
    artifact_path : str, optional, default="model"
        The path of the model artifact in the MLflow run.

    Returns
    -------
    mlflow.sklearn.PyFuncModel
        The loaded model.
    """

    model_uri = f"runs:/{run_id}/{artifact_path}"

    # Load and return the model
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded successfully from run ID {run_id} at path {artifact_path}")
        return model
    except Exception as e:
        print(f"Error loading model from MLflow: {e}")
        return None


def model_validation(
    estimator: ClassifierMixin,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    estimator_name: str,
) -> pd.DataFrame:
    """
    Perform cross-validation on a given classifier to evaluate its performance.

    This function uses Stratified K-Fold cross-validation to assess the performance of the provided estimator
    on the training data. The cross-validation scores are computed using a custom scoring function.

    Parameters
    ----------
    estimator : (ClassifierMixin)
        The classifier to be evaluated. This should be an instance of a scikit-learn classifier that implements the `fit` and `predict` methods.
    X_train : (pd.DataFrame)
        The feature set used for training the estimator. It should be a DataFrame with shape (n_samples, n_features).
    y_train : (pd.Series)
        The target labels corresponding to the training features. It should be a Series with shape (n_samples,).
    estimator_name : (str)
        A string representing the name of the estimator. This is used for labeling the results in the output DataFrame.

    Returns
    -------
        pd.DataFrame: A DataFrame containing the cross-validation scores for each fold, with the
        model name as the first column. The DataFrame has the following structure:
    """

    stratified_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cross_val_scores = cross_val_score(
        estimator,
        X_train,
        y_train,
        cv=stratified_cv,
        scoring=custom_scorer(),
    )

    results = pd.DataFrame(
        data=[cross_val_scores],
        columns=["1st fold", "2nd fold", "3rd fold", "4th fold", "5th fold"],
    )

    results.insert(0, "Model", estimator_name)
    return results


def model_prediction(
    estimator: ClassifierMixin,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """
    Makes predictions for a binary classification model based on a custom decision threshold.

    Parameters
    ----------
    estimator : ClassifierMixin
        A trained classification model that has the `predict_proba` method for predicting probabilities, or a PyFuncModel which has only `predict`.
    X_test : pd.DataFrame
        Test data containing feature variables for which predictions need to be made.

    Returns
    -------
    np.ndarray
        A NumPy array of binary predictions (0 or 1), where the decision threshold is applied to the predicted probabilities (if available).

    Notes
    ------
    - The method uses `predict_proba` to obtain predicted probabilities for the positive class (class 1).
    - A custom decision threshold of 0.4 is applied to determine the final binary prediction.
    - Predictions with a probability greater than or equal to 0.4 are classified as class 1, otherwise class 0.
    """

    if hasattr(estimator, "predict_proba"):
        y_pred_prob = estimator.predict_proba(X_test)[:, 1]
    else:
        y_pred_prob = estimator.predict(X_test)

        if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 2:
            y_pred_prob = y_pred_prob[:, 1]
        elif y_pred_prob.ndim == 1:
            return y_pred_prob

    decision_threshold = 0.4
    y_pred = (y_pred_prob >= decision_threshold).astype(int)
    return y_pred


def model_confusion_matrix(
    estimator: ClassifierMixin,
    y_test: pd.Series,
    y_pred: np.ndarray,
    ax: plt.Axes,
) -> None:
    """
    Plots the confusion matrix for a given model's predictions.

    Parameters
    ----------
    model : ClassifierMixin
        The trained classification model which has the `classes_` attribute representing the class labels.
    y_test : pd.Series
        The true labels for the test dataset.
    y_pred : np.ndarray
        The predicted labels generated by the model.
    ax : plt.Axes
        The matplotlib Axes on which to plot the confusion matrix.

    Returns
    -------
    None
        This function doesn't return any value but displays the confusion matrix plot on the provided Axes.

    Notes
    ------
    - The confusion matrix is computed using the true and predicted labels (`y_test` and `y_pred`).
    - The model's `classes_` attribute is used to ensure the correct order of class labels.
    - The confusion matrix is displayed using `ConfusionMatrixDisplay` for clearer visualization.
    """

    cm = confusion_matrix(y_test, y_pred, labels=estimator.classes_)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=estimator.classes_)
    cm_display.plot(ax=ax)


def model_classification_report(
    estimator_predictions: Dict[str, Any],
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Generates a classification report for multiple models and returns the results in a DataFrame.

    Parameters
    ----------
    estimator_predictions : Dict[str, Any]
        A dictionary where the keys are model names (strings) and the values are the predicted labels (np.ndarray) for the test set from each model.
    y_test : pd.Series
        The true labels for the test dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing model performance metrics for each model, including Accuracy, Precision, Recall, and F1 Score for the positive class (`1.0`).

    Notes
    ------
    - The function iterates over the dictionary of model predictions, computes the classification report for each model, and stores key metrics.
    - The output includes the Accuracy, Precision, Recall, and F1 Score specifically for the positive class (`1.0`).
    - The function returns a pandas DataFrame summarizing these metrics for all the models.
    """

    classification_report_data: List[Dict[str, Any]] = []
    for model_name, model_pred in estimator_predictions.items():
        report = classification_report(y_test, model_pred, output_dict=True)
        classification_report_data.append(
            {
                "Model": model_name,
                "Accuracy Score": report["accuracy"],
                "Precision Score": report["1.0"]["precision"],
                "Recall Score": report["1.0"]["recall"],
                "F1 Score": report["1.0"]["f1-score"],
            }
        )

    classification_report_df = pd.DataFrame(classification_report_data)
    return classification_report_df
