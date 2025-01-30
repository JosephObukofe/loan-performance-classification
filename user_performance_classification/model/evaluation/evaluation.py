import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from user_performance_classification.data.custom import (
    load_data_from_minio,
    load_model_from_mlflow,
    model_prediction,
    model_confusion_matrix,
    model_classification_report,
)

from user_performance_classification.config.config import (
    MLFLOW_TRACKING_URI,
    CONFUSION_MATRIX,
    MODEL_EVALUATION_REPORT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

mlflow.set_experiment("Loan Performance Classification: Model Evaluation")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Load trained models from MLflow
logistic_regression_model_transformed = load_model_from_mlflow(
    run_id="41d7302050754a95b5f416098f1f8a63",
    artifact_path="loan_performance_logistic_regression_classifier_20250129_154746",
)

logistic_regression_model_binned_encoded = load_model_from_mlflow(
    run_id="2a7819ac9bf9431aaf643f52fdd8a5d1",
    artifact_path="loan_performance_logistic_regression_classifier_20250129_154750",
)

logistic_regression_model_binned_labeled = load_model_from_mlflow(
    run_id="643990d09de94d87a013575873deef97",
    artifact_path="loan_performance_logistic_regression_classifier_20250129_154741",
)


decision_tree_model_transformed = load_model_from_mlflow(
    run_id="17e46353252f4b3ea6a99ccede8fb2c1",
    artifact_path="loan_performance_decision_tree_classifier_20250129_154802",
)

decision_tree_model_binned_encoded = load_model_from_mlflow(
    run_id="793ac9ed0d064c6e9b336e9526f86256",
    artifact_path="loan_performance_decision_tree_classifier_20250129_154759",
)

decision_tree_model_binned_labeled = load_model_from_mlflow(
    run_id="6f8530b4dd0d4e0ca266b00379925289",
    artifact_path="loan_performance_decision_tree_classifier_20250129_154803",
)

random_forest_model_transformed = load_model_from_mlflow(
    run_id="792f5fdf79834352ad656e1a2bb86876",
    artifact_path="loan_performance_random_forest_classifier_20250129_155946",
)

random_forest_model_binned_encoded = load_model_from_mlflow(
    run_id="a16d3bfdd9f94145a84a8189c0b74639",
    artifact_path="loan_performance_random_forest_classifier_20250129_155928",
)

random_forest_model_binned_labeled = load_model_from_mlflow(
    run_id="6226aa6945484ff0a817b22072ffe4b8",
    artifact_path="loan_performance_random_forest_classifier_20250129_155949",
)

svc_model_transformed = load_model_from_mlflow(
    run_id="fa4a84dd2649417a8131c38a182713bd",
    artifact_path="loan_performance_support_vector_classifier_20250129_155015",
)

svc_model_binned_encoded = load_model_from_mlflow(
    run_id="e1b0b07fb47a4b51b6e6504e1da60831",
    artifact_path="loan_performance_support_vector_classifier_20250129_155009",
)

svc_model_binned_labeled = load_model_from_mlflow(
    run_id="af0e78f2f76749e5bcb6595125a96ec4",
    artifact_path="loan_performance_support_vector_classifier_20250129_155044",
)

hgboost_model_transformed = load_model_from_mlflow(
    run_id="0fecdf59a3604abc816ab43605b431b4",
    artifact_path="loan_performance_histogram_gradient_boosting_classifier_20250129_155501",
)

hgboost_model_binned_encoded = load_model_from_mlflow(
    run_id="a77c2108e2c24631ae8752d7823ce6be",
    artifact_path="loan_performance_histogram_gradient_boosting_classifier_20250129_155521",
)

hgboost_model_binned_labeled = load_model_from_mlflow(
    run_id="c9cf1a3d97c94c93b1e381b8b9fab19b",
    artifact_path="loan_performance_histogram_gradient_boosting_classifier_20250129_155512",
)


# Load datasets from MinIO
# Numerically Transformed Dataset
X_test_transformed = load_data_from_minio(
    bucket_name="processed",
    object_name="X_test_transformed.pkl",
)

y_test_transformed = load_data_from_minio(
    bucket_name="processed",
    object_name="y_test_transformed.pkl",
)

# Discretized and Encoded Dataset
X_test_binned_encoded = load_data_from_minio(
    bucket_name="processed",
    object_name="X_test_binned_encoded.pkl",
)

y_test_binned_encoded = load_data_from_minio(
    bucket_name="processed",
    object_name="y_test_binned_encoded.pkl",
)

# Discretized and Labeled Dataset
X_test_binned_labeled = load_data_from_minio(
    bucket_name="processed",
    object_name="X_test_binned_labeled.pkl",
)

y_test_binned_labeled = load_data_from_minio(
    bucket_name="processed",
    object_name="y_test_binned_labeled.pkl",
)


# Define datasets and their corresponding test labels
test_datasets = {
    "transformed": (X_test_transformed, y_test_transformed),
    "binned_encoded": (X_test_binned_encoded, y_test_binned_encoded),
    "binned_labeled": (X_test_binned_labeled, y_test_binned_labeled),
}

models = {
    "Logistic Regression with transformed dataset": logistic_regression_model_transformed,
    "Logistic Regression with binned_encoded dataset": logistic_regression_model_binned_encoded,
    "Logistic Regression with binned_labeled dataset": logistic_regression_model_binned_labeled,
    "Decision Tree with transformed dataset": decision_tree_model_transformed,
    "Decision Tree with binned_encoded dataset": decision_tree_model_binned_encoded,
    "Decision Tree with binned_labeled dataset": decision_tree_model_binned_labeled,
    "Random Forest with transformed dataset": random_forest_model_transformed,
    "Random Forest with binned_encoded dataset": random_forest_model_binned_encoded,
    "Random Forest with binned_labeled dataset": random_forest_model_binned_labeled,
    "SVM with transformed dataset": svc_model_transformed,
    "SVM with binned_encoded dataset": svc_model_binned_encoded,
    "SVM with binned_labeled dataset": svc_model_binned_labeled,
    "HGBoost with transformed dataset": hgboost_model_transformed,
    "HGBoost with binned_encoded dataset": hgboost_model_binned_encoded,
    "HGBoost with binned_labeled dataset": hgboost_model_binned_labeled,
}

model_predictions = {}

for model_name, model in models.items():
    for dataset_name, (X_test, y_test) in test_datasets.items():
        key = f"{model_name} with {dataset_name} dataset"
        model_predictions[key] = model_prediction(model, X_test)


# Start MLflow Run
with mlflow.start_run(run_name="Model Evaluation Operation Run"):

    fig, axes = plt.subplots(
        len(models),  # 15 models
        len(test_datasets),  # 3 datasets
        figsize=(15, 12),
    )
    axes = axes.flatten()

    for i, (model_dataset_name, y_pred) in enumerate(model_predictions.items()):
        model_name, dataset_name = model_dataset_name.split(" with ")
        X_test, y_test = test_datasets[dataset_name]

        # Compute Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log metrics in MLflow
        mlflow.log_metrics(
            {
                f"{model_dataset_name} Accuracy": accuracy,
                f"{model_dataset_name} Precision": precision,
                f"{model_dataset_name} Recall": recall,
                f"{model_dataset_name} F1": f1,
            }
        )

        # Generate and log confusion matrix
        model_confusion_matrix(models[model_dataset_name], y_test, y_pred, ax=axes[i])
        axes[i].set_title(model_dataset_name)

    # Save and log confusion matrices
    plt.tight_layout()
    confusion_matrix_path = CONFUSION_MATRIX
    plt.savefig(confusion_matrix_path)

    try:
        mlflow.log_artifact(
            confusion_matrix_path,
            artifact_path="confusion_matrices",
        )
    except Exception as e:
        logger.error(f"Error logging confusion matrix artifact: {e}")

    os.remove(confusion_matrix_path)

    # Generate and log model evaluation report
    model_evaluation_report = model_classification_report(
        model_predictions,
        test_datasets,
    )
    model_evaluation_report_path = MODEL_EVALUATION_REPORT
    model_evaluation_report.to_csv(
        model_evaluation_report_path,
        index=False,
    )

    try:
        mlflow.log_artifact(
            model_evaluation_report_path,
            artifact_path="evaluation_reports",
        )
    except Exception as e:
        logger.error(f"Error logging evaluation report artifact: {e}")

    os.remove(model_evaluation_report_path)
