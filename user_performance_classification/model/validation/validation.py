import os
import joblib
import numpy as np
import pandas as pd
import mlflow
from user_performance_classification.data.custom import (
    load_data_from_minio,
    load_model_from_mlflow,
    model_validation,
)
from user_performance_classification.config.config import (
    MLFLOW_TRACKING_URI,
    MODEL_VALIDATION_REPORT,
)

mlflow.set_experiment("Loan Performance Classification: Model Validation")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Load datasets from MinIO
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


# Loading the models from MLflow
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


with mlflow.start_run(run_name="Model Validation Operation Run"):
    logistic_regressor_model_validation_transformed = model_validation(
        logistic_regression_model_transformed,
        X_train_transformed,
        y_train_transformed,
        "Logistic Regression Classifier (Transformed)",
    )

    logistic_regressor_model_validation_binned_encoded = model_validation(
        logistic_regression_model_binned_encoded,
        X_train_binned_encoded,
        y_train_binned_encoded,
        "Logistic Regression Classifier (Binned Encoded)",
    )

    logistic_regressor_model_validation_binned_labeled = model_validation(
        logistic_regression_model_binned_labeled,
        X_train_binned_labeled,
        y_train_binned_labeled,
        "Logistic Regression Classifier (Binned Labeled)",
    )

    decision_tree_model_validation_transformed = model_validation(
        decision_tree_model_transformed,
        X_train_transformed,
        y_train_transformed,
        "Decision Tree Classifier (Transformed)",
    )

    decision_tree_model_validation_binned_encoded = model_validation(
        decision_tree_model_binned_encoded,
        X_train_binned_encoded,
        y_train_binned_encoded,
        "Decision Tree Classifier (Binned Encoded)",
    )

    decision_tree_model_validation_binned_labeled = model_validation(
        decision_tree_model_binned_labeled,
        X_train_binned_labeled,
        y_train_binned_labeled,
        "Decision Tree Classifier (Binned Labeled)",
    )

    random_forest_model_validation_transformed = model_validation(
        random_forest_model_transformed,
        X_train_transformed,
        y_train_transformed,
        "Random Forest Classifier (Transformed)",
    )

    random_forest_model_validation_binned_encoded = model_validation(
        random_forest_model_binned_encoded,
        X_train_binned_encoded,
        y_train_binned_encoded,
        "Random Forest Classifier (Binned Encoded)",
    )

    random_forest_model_validation_binned_labeled = model_validation(
        random_forest_model_binned_labeled,
        X_train_binned_labeled,
        y_train_binned_labeled,
        "Random Forest Classifier (Binned Labeled)",
    )

    svc_model_validation_transformed = model_validation(
        svc_model_transformed,
        X_train_transformed,
        y_train_transformed,
        "SVC Classifier (Transformed)",
    )

    svc_model_validation_binned_encoded = model_validation(
        svc_model_binned_encoded,
        X_train_binned_encoded,
        y_train_binned_encoded,
        "SVC Classifier (Binned Encoded)",
    )

    svc_model_validation_binned_labeled = model_validation(
        svc_model_binned_labeled,
        X_train_binned_labeled,
        y_train_binned_labeled,
        "SVC Classifier (Binned Labeled)",
    )

    hgboost_model_validation_transformed = model_validation(
        hgboost_model_transformed,
        X_train_transformed,
        y_train_transformed,
        "HGBoost Classifier (Transformed)",
    )

    hgboost_model_validation_binned_encoded = model_validation(
        hgboost_model_binned_encoded,
        X_train_binned_encoded,
        y_train_binned_encoded,
        "HGBoost Classifier (Binned Encoded)",
    )

    hgboost_model_validation_binned_labeled = model_validation(
        hgboost_model_binned_labeled,
        X_train_binned_labeled,
        y_train_binned_labeled,
        "HGBoost Classifier (Binned Labeled)",
    )

    combined_validation = pd.concat(
        [
            logistic_regressor_model_validation_transformed,
            logistic_regressor_model_validation_binned_encoded,
            logistic_regressor_model_validation_binned_labeled,
            decision_tree_model_validation_transformed,
            decision_tree_model_validation_binned_encoded,
            decision_tree_model_validation_binned_labeled,
            random_forest_model_validation_transformed,
            random_forest_model_validation_binned_encoded,
            random_forest_model_validation_binned_labeled,
            svc_model_validation_transformed,
            svc_model_validation_binned_encoded,
            svc_model_validation_binned_labeled,
            hgboost_model_validation_transformed,
            hgboost_model_validation_binned_encoded,
            hgboost_model_validation_binned_labeled,
        ],
        axis=0,
        ignore_index=True,
    )

    print(combined_validation)

    combined_validation_path = MODEL_VALIDATION_REPORT
    combined_validation.to_csv(combined_validation_path, index=False)
    mlflow.log_artifact(
        combined_validation_path,
        artifact_path="validation_reports",
    )
    os.remove(combined_validation_path)
