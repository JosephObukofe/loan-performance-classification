import numpy as np
import pandas as pd
import joblib
import mlflow
import logging
from skopt.space import Categorical, Integer, Real
from joblib import Parallel, delayed
from user_performance_classification.data.custom import (
    load_data_from_minio,
    train_logistic_regression_classifier,
    train_decision_tree_classifier,
    train_random_forest_classifier,
    train_svm_classifier,
    train_hgboost_classifier,
)
from user_performance_classification.config.config import MLFLOW_TRACKING_URI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

mlflow.set_experiment("Loan Performance Classification: Model Training")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Numerically Transformed Dataset
X_train_transformed = load_data_from_minio(
    bucket_name="processed",
    object_name="X_train_transformed.pkl",
)

y_train_transformed = load_data_from_minio(
    bucket_name="processed",
    object_name="y_train_transformed.pkl",
)

# Discretized and Encoded Dataset
X_train_binned_encoded = load_data_from_minio(
    bucket_name="processed",
    object_name="X_train_binned_encoded.pkl",
)

y_train_binned_encoded = load_data_from_minio(
    bucket_name="processed",
    object_name="y_train_binned_encoded.pkl",
)

# Discretized and Labeled Dataset
X_train_binned_labeled = load_data_from_minio(
    bucket_name="processed",
    object_name="X_train_binned_labeled.pkl",
)

y_train_binned_labeled = load_data_from_minio(
    bucket_name="processed",
    object_name="y_train_binned_labeled.pkl",
)

logistic_regression_model_search_space = {
    "C": Real(0.01, 10),
    "penalty": Categorical(["l1", "l2", "elasticnet"]),
    "max_iter": Integer(100, 500),
    "l1_ratio": Real(0.1, 0.9),
}

decision_tree_model_search_space = {
    "criterion": Categorical(["gini", "entropy", "log_loss"]),
    "max_depth": Integer(5, 50),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 10),
    "max_features": Categorical(["sqrt", "log2"]),
    "splitter": Categorical(["best", "random"]),
}

random_forest_model_search_space = {
    "n_estimators": Integer(100, 500),
    "max_features": Categorical(["sqrt", "log2"]),
    "max_depth": Integer(10, 100),
    "min_samples_split": Integer(2, 10),
    "min_samples_leaf": Integer(1, 10),
}

svc_model_search_space = {
    "C": Real(0.1, 100),
    "kernel": Categorical(["linear", "rbf", "poly", "sigmoid"]),
    "gamma": Categorical(["scale", "auto"]),
    "degree": Integer(2, 4),
    "max_iter": Integer(-1, 5000),
}

hgboost_model_search_space = {
    "learning_rate": Real(0.01, 0.3),
    "max_iter": Integer(100, 500),
    "max_depth": Integer(5, 20),
    "min_samples_leaf": Integer(10, 50),
    "l2_regularization": Real(0, 10),
    "max_bins": Integer(2, 255),
    "scoring": Categorical(["accuracy", "f1", "roc_auc"]),
}


def safe_task(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
        return None


def parallel_model_training():
    tasks = [
        delayed(safe_task)(
            train_logistic_regression_classifier,
            X_train_transformed,
            y_train_transformed,
            logistic_regression_model_search_space,
            "transformed",
        ),
        delayed(safe_task)(
            train_logistic_regression_classifier,
            X_train_binned_encoded,
            y_train_binned_encoded,
            logistic_regression_model_search_space,
            "binned_encoded",
        ),
        delayed(safe_task)(
            train_logistic_regression_classifier,
            X_train_binned_labeled,
            y_train_binned_labeled,
            logistic_regression_model_search_space,
            "binned_labeled",
        ),
        delayed(safe_task)(
            train_decision_tree_classifier,
            X_train_transformed,
            y_train_transformed,
            decision_tree_model_search_space,
            "transformed",
        ),
        delayed(safe_task)(
            train_decision_tree_classifier,
            X_train_binned_encoded,
            y_train_binned_encoded,
            decision_tree_model_search_space,
            "binned_encoded",
        ),
        delayed(safe_task)(
            train_decision_tree_classifier,
            X_train_binned_labeled,
            y_train_binned_labeled,
            decision_tree_model_search_space,
            "binned_labeled",
        ),
        delayed(safe_task)(
            train_random_forest_classifier,
            X_train_transformed,
            y_train_transformed,
            random_forest_model_search_space,
            "transformed",
        ),
        delayed(safe_task)(
            train_random_forest_classifier,
            X_train_binned_encoded,
            y_train_binned_encoded,
            random_forest_model_search_space,
            "binned_encoded",
        ),
        delayed(safe_task)(
            train_random_forest_classifier,
            X_train_binned_labeled,
            y_train_binned_labeled,
            random_forest_model_search_space,
            "binned_labeled",
        ),
        delayed(safe_task)(
            train_svm_classifier,
            X_train_transformed,
            y_train_transformed,
            svc_model_search_space,
            "transformed",
        ),
        delayed(safe_task)(
            train_svm_classifier,
            X_train_binned_encoded,
            y_train_binned_encoded,
            svc_model_search_space,
            "binned_encoded",
        ),
        delayed(safe_task)(
            train_svm_classifier,
            X_train_binned_labeled,
            y_train_binned_labeled,
            svc_model_search_space,
            "binned_labeled",
        ),
        delayed(safe_task)(
            train_hgboost_classifier,
            X_train_transformed,
            y_train_transformed,
            hgboost_model_search_space,
            "transformed",
        ),
        delayed(safe_task)(
            train_hgboost_classifier,
            X_train_binned_encoded,
            y_train_binned_encoded,
            hgboost_model_search_space,
            "binned_encoded",
        ),
        delayed(safe_task)(
            train_hgboost_classifier,
            X_train_binned_labeled,
            y_train_binned_labeled,
            hgboost_model_search_space,
            "binned_labeled",
        ),
    ]

    trained_models = Parallel(n_jobs=6, verbose=10)(tasks)
    return trained_models


# Training each model in parallel
if __name__ == "__main__":
    trained_models_output = parallel_model_training()
