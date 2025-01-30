import shap
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from skopt.space import Categorical, Integer, Real
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from user_performance_classification.data.custom import (
    read_csv_from_minio,
    preprocess_training_data,
    preprocess_test_data,
    bayesian_hyperparameter_tuning,
)

df = read_csv_from_minio(
    bucket_name="data",
    object_name="classification_data.csv",
)

# Renaming the columns to make them more structured
df_renamed = df.rename(
    columns={
        "Credit Amount": "credit_amount",
        "Age": "age",
        "Duration of Credit (month)": "duration_of_credit_in_months",
        "Purpose": "purpose",
        "Payment Status of Previous Credit": "payment_status_of_previous_credit",
        "Savings type": "savings_type",
        "Length of current employment": "length_of_current_employment",
        "Account type": "account_type",
        "Instalment per cent": "instalment_per_cent",
        "Marital Status": "marital_status",
        "Occupation": "occupation",
        "Duration in Current address": "duration_in_current_address",
        "Most valuable available asset": "most_valuable_available_asset",
        "No of Credits at this Bank": "number_of_credits_at_this_bank",
        "Type of apartment": "apartment_type",
        "Guarantors": "number_of_guarantors",
        "Concurrent Credits": "number_of_concurrent_credits",
        "label": "loan_performance",
        "No of dependents": "number_of_dependents",
        "Telephone": "telephone",
        "Foreign Worker": "foreign_worker",
    }
)

# Binning continuous variables for df_binned
age_bins = [
    0,
    20,
    30,
    40,
    50,
    60,
    100,
]

age_labels = [
    "<20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60+",
]

credit_amount_bins = [
    0,
    1000,
    2000,
    3000,
    4000,
    5000,
    10000,
    12000,
    14000,
    16000,
    18000,
    20000,
]

credit_amount_labels = [
    "<1000",
    "1000-2000",
    "2000-3000",
    "3000-4000",
    "4000-5000",
    "5000-10000",
    "10000-12000",
    "12000-14000",
    "14000-16000",
    "16000-18000",
    "18000+",
]

duration_of_credit_bins = [
    0,
    12,
    24,
    36,
    48,
    100,
]

duration_of_credit_labels = [
    "<12",
    "12-24",
    "24-36",
    "36-48",
    "48+",
]

df_binned = df_renamed.copy()
df_transformed = df_renamed.copy()

df_binned["age_binned"] = pd.cut(
    df_binned["age"],
    bins=age_bins,
    labels=age_labels,
    right=False,
)

df_binned["credit_amount_binned"] = pd.cut(
    df_binned["credit_amount"],
    bins=credit_amount_bins,
    labels=credit_amount_labels,
    right=False,
)

df_binned["duration_of_credit_in_months_binned"] = pd.cut(
    df_binned["duration_of_credit_in_months"],
    bins=duration_of_credit_bins,
    labels=duration_of_credit_labels,
    right=False,
)


# Implementing ordinal encoding for the binned features
age_ordinal_mapping = {k: i for i, k in enumerate(age_labels)}
credit_amount_ordinal_mapping = {k: i for i, k in enumerate(credit_amount_labels)}
duration_of_credit_ordinal_mapping = {
    k: i for i, k in enumerate(duration_of_credit_labels)
}

df_binned["age_binned_encoded"] = df_binned["age_binned"].map(age_ordinal_mapping)
df_binned["credit_amount_binned_encoded"] = df_binned["credit_amount_binned"].map(
    credit_amount_ordinal_mapping
)
df_binned["duration_of_credit_in_months_binned_encoded"] = df_binned[
    "duration_of_credit_in_months_binned"
].map(duration_of_credit_ordinal_mapping)


# Dropping the original features to prevent multicollinearity issues
df_binned.drop(
    columns=[
        "age",
        "age_binned",
        "credit_amount",
        "credit_amount_binned",
        "duration_of_credit_in_months",
        "duration_of_credit_in_months_binned",
    ],
    inplace=True,
)


# Dropping the "user_id" column as it is a flat and wide feature that does not provide any information
df_binned.drop("user_id", axis=1, inplace=True)
df_transformed.drop("user_id", axis=1, inplace=True)


# Creating a copy of the binned dataframe for the labeled and one-hot encoding transformations
df_binned_labeled = df_binned.copy()
df_binned_encoded = df_binned.copy()


target = "loan_performance"
transformed_features = df_transformed.columns[df_transformed.columns != target]
binned_labeled_features = df_binned_labeled.columns[df_binned_labeled.columns != target]
binned_encoded_features = df_binned_encoded.columns[df_binned_encoded.columns != target]

transformed_numerical_features = [
    "credit_amount",
    "duration_of_credit_in_months",
    "age",
]

transformed_categorical_features = [
    feature
    for feature in transformed_features
    if feature not in transformed_numerical_features
]

binned_labeled_numerical_features = [
    "credit_amount_binned_encoded",
    "duration_of_credit_in_months_binned_encoded",
    "age_binned_encoded",
]

binned_labeled_categorical_features = [
    feature
    for feature in binned_labeled_features
    if feature not in binned_labeled_numerical_features
]

binned_encoded_numerical_features = [
    "credit_amount_binned_encoded",
    "duration_of_credit_in_months_binned_encoded",
    "age_binned_encoded",
]

binned_encoded_categorical_features = [feature for feature in binned_encoded_features]


# Casting the numerical features to float64 and the categorical features to category for the transformed dataframe
df_transformed[transformed_numerical_features] = df_transformed[
    transformed_numerical_features
].astype("float64")

df_transformed[transformed_categorical_features] = df_transformed[
    transformed_categorical_features
].astype("category")


# Casting the numerical features to category for the binned labeled dataframe
df_binned_labeled[binned_labeled_numerical_features] = df_binned_labeled[
    binned_labeled_numerical_features
].astype("category")

df_binned_labeled[binned_labeled_categorical_features] = df_binned_labeled[
    binned_labeled_categorical_features
].astype("category")


# Casting the numerical features to category for the binned encoded dataframe
df_binned_encoded[binned_encoded_numerical_features] = df_binned_encoded[
    binned_encoded_numerical_features
].astype("category")

df_binned_encoded[binned_encoded_categorical_features] = df_binned_encoded[
    binned_encoded_categorical_features
].astype("category")


# Splitting the datasets into training and testing sets
X_transformed = df_transformed[transformed_features]
y_transformed = df_transformed[target]
X_binned_labeled = df_binned_labeled[binned_labeled_features]
y_binned_labeled = df_binned_labeled[target]
X_binned_encoded = df_binned_encoded[binned_encoded_features]
y_binned_encoded = df_binned_encoded[target]

(
    X_transformed_train,
    X_transformed_test,
    y_transformed_train,
    y_transformed_test,
) = train_test_split(
    X_transformed,
    y_transformed,
    test_size=0.2,
    stratify=y_transformed,
    random_state=42,
)

(
    X_binned_labeled_train,
    X_binned_labeled_test,
    y_binned_labeled_train,
    y_binned_labeled_test,
) = train_test_split(
    X_binned_labeled,
    y_binned_labeled,
    test_size=0.2,
    stratify=y_binned_labeled,
    random_state=43,
)

(
    X_binned_encoded_train,
    X_binned_encoded_test,
    y_binned_encoded_train,
    y_binned_encoded_test,
) = train_test_split(
    X_binned_encoded,
    y_binned_encoded,
    test_size=0.2,
    stratify=y_binned_encoded,
    random_state=44,
)


# Preprocessing the training and testing datasets
# For the transformed dataset
(
    X_transformed_train_preprocessed,
    transforming_pipeline,
    transformed_column_names,
) = preprocess_training_data(
    X=X_transformed_train,
    numeric_features=transformed_numerical_features,
    categorical_features=transformed_categorical_features,
    encoding_type="box-cox",
)

X_transformed_test_preprocessed = preprocess_test_data(
    X=X_transformed_test,
    pipeline=transforming_pipeline,
    processed_column_names=transformed_column_names,
)


# For the binned labeled dataset
(
    X_binned_labeled_train_preprocessed,
    binned_labeled_pipeline,
    binned_labeled_column_names,
) = preprocess_training_data(
    X=X_binned_labeled_train,
    numeric_features=binned_labeled_numerical_features,
    categorical_features=binned_labeled_categorical_features,
    encoding_type="none",
)

X_binned_labeled_test_preprocessed = preprocess_test_data(
    X=X_binned_labeled_test,
    pipeline=binned_labeled_pipeline,
    processed_column_names=binned_labeled_column_names,
)


# For the binned encoded dataset
(
    X_binned_encoded_train_preprocessed,
    binned_encoded_pipeline,
    binned_encoded_column_names,
) = preprocess_training_data(
    X=X_binned_encoded_train,
    numeric_features=binned_encoded_numerical_features,
    categorical_features=binned_encoded_categorical_features,
    encoding_type="none",
)

X_binned_encoded_test_preprocessed = preprocess_test_data(
    X=X_binned_encoded_test,
    pipeline=binned_encoded_pipeline,
    processed_column_names=binned_encoded_column_names,
)


# Oversampling the "train_preprocessed" datasets using SMOTE
smote = SMOTE(random_state=42)
X_transformed_train_preprocessed_balanced, y_transformed_train_balanced = (
    smote.fit_resample(
        X_transformed_train_preprocessed,
        y_transformed_train,
    )
)

X_binned_labeled_train_preprocessed_balanced, y_binned_labeled_train_balanced = (
    smote.fit_resample(
        X_binned_labeled_train_preprocessed,
        y_binned_labeled_train,
    )
)

X_binned_encoded_train_preprocessed_balanced, y_binned_encoded_train_balanced = (
    smote.fit_resample(
        X_binned_encoded_train_preprocessed,
        y_binned_encoded_train,
    )
)

# Logistic Regression Classifier
log_reg_feature_eval_model = LogisticRegression(
    max_iter=10000,
    class_weight={0: 1, 1: 4},
    random_state=42,
)

log_reg_feature_eval_search_space = {
    "C": Real(0.01, 10),
    "penalty": ["l1"],
}

log_reg_feature_eval_tuned_model_transformed = bayesian_hyperparameter_tuning(
    estimator=log_reg_feature_eval_model,
    search_spaces=log_reg_feature_eval_search_space,
    X_train=X_transformed_train_preprocessed_balanced,
    y_train=y_transformed_train_balanced,
)

log_reg_feature_eval_tuned_model_binned_labeled = bayesian_hyperparameter_tuning(
    estimator=log_reg_feature_eval_model,
    search_spaces=log_reg_feature_eval_search_space,
    X_train=X_binned_labeled_train_preprocessed_balanced,
    y_train=y_binned_labeled_train_balanced,
)

log_reg_feature_eval_tuned_model_binned_encoded = bayesian_hyperparameter_tuning(
    estimator=log_reg_feature_eval_model,
    search_spaces=log_reg_feature_eval_search_space,
    X_train=X_binned_encoded_train_preprocessed_balanced,
    y_train=y_binned_encoded_train_balanced,
)

log_reg_feature_eval_tuned_model_transformed.fit(
    X_transformed_train_preprocessed_balanced,
    y_transformed_train_balanced,
)

log_reg_feature_eval_tuned_model_binned_labeled.fit(
    X_binned_labeled_train_preprocessed_balanced,
    y_binned_labeled_train_balanced,
)

log_reg_feature_eval_tuned_model_binned_encoded.fit(
    X_binned_encoded_train_preprocessed_balanced,
    y_binned_encoded_train_balanced,
)


# SHAP Explainers for the Logistic Regression models
log_reg_feature_eval_tuned_model_shap_explainer_transformed = shap.LinearExplainer(
    log_reg_feature_eval_tuned_model_transformed,
    X_transformed_train_preprocessed_balanced,
)

log_reg_feature_eval_tuned_model_shap_explainer_binned_labeled = shap.LinearExplainer(
    log_reg_feature_eval_tuned_model_binned_labeled,
    X_binned_labeled_train_preprocessed_balanced,
)

log_reg_feature_eval_tuned_model_shap_explainer_binned_encoded = shap.LinearExplainer(
    log_reg_feature_eval_tuned_model_binned_encoded,
    X_binned_encoded_train_preprocessed_balanced,
)


# SHAP Values for the Logistic Regression models
log_reg_feature_eval_tuned_model_shap_values_transformed = (
    log_reg_feature_eval_tuned_model_shap_explainer_transformed.shap_values(
        X_transformed_train_preprocessed_balanced
    )
)

log_reg_feature_eval_tuned_model_shap_values_binned_labeled = (
    log_reg_feature_eval_tuned_model_shap_explainer_binned_labeled.shap_values(
        X_binned_labeled_train_preprocessed_balanced
    )
)

log_reg_feature_eval_tuned_model_shap_values_binned_encoded = (
    log_reg_feature_eval_tuned_model_shap_explainer_binned_encoded.shap_values(
        X_binned_encoded_train_preprocessed_balanced
    )
)

# Mean absolute SHAP values for each feature for the Logistic Regression models
mean_abs_log_reg_feature_eval_tuned_model_shap_values_transformed = np.mean(
    np.abs(log_reg_feature_eval_tuned_model_shap_values_transformed),
    axis=0,
)

mean_abs_log_reg_feature_eval_tuned_model_shap_values_transformed_df = (
    pd.DataFrame(
        {
            "Feature": X_transformed_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": mean_abs_log_reg_feature_eval_tuned_model_shap_values_transformed,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)

mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_labeled = np.mean(
    np.abs(log_reg_feature_eval_tuned_model_shap_values_binned_labeled),
    axis=0,
)

mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_labeled_df = (
    pd.DataFrame(
        {
            "Feature": X_binned_labeled_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_labeled,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)

mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_encoded = np.mean(
    np.abs(log_reg_feature_eval_tuned_model_shap_values_binned_encoded),
    axis=0,
)

mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_encoded_df = (
    pd.DataFrame(
        {
            "Feature": X_binned_encoded_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_encoded,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)


# Random Forest Classifier
rand_for_feature_eval_model = RandomForestClassifier(
    class_weight={0: 1, 1: 4},
    random_state=42,
)

rand_for_feature_eval_search_space = {
    "n_estimators": Integer(100, 1000),
    "max_depth": Integer(1, 50),
    "min_samples_split": Integer(2, 6),
    "min_samples_leaf": Integer(1, 4),
    "criterion": Categorical(["gini", "entropy"]),
}

rand_for_feature_eval_tuned_model_transformed = bayesian_hyperparameter_tuning(
    estimator=rand_for_feature_eval_model,
    search_spaces=rand_for_feature_eval_search_space,
    X_train=X_transformed_train_preprocessed_balanced,
    y_train=y_transformed_train_balanced,
)

rand_for_feature_eval_tuned_model_binned_labeled = bayesian_hyperparameter_tuning(
    estimator=rand_for_feature_eval_model,
    search_spaces=rand_for_feature_eval_search_space,
    X_train=X_binned_labeled_train_preprocessed_balanced,
    y_train=y_binned_labeled_train_balanced,
)

rand_for_feature_eval_tuned_model_binned_encoded = bayesian_hyperparameter_tuning(
    estimator=rand_for_feature_eval_model,
    search_spaces=rand_for_feature_eval_search_space,
    X_train=X_binned_encoded_train_preprocessed_balanced,
    y_train=y_binned_encoded_train_balanced,
)

rand_for_feature_eval_tuned_model_transformed.fit(
    X_transformed_train_preprocessed_balanced,
    y_transformed_train_balanced,
)

rand_for_feature_eval_tuned_model_binned_labeled.fit(
    X_binned_labeled_train_preprocessed_balanced,
    y_binned_labeled_train_balanced,
)

rand_for_feature_eval_tuned_model_binned_encoded.fit(
    X_binned_encoded_train_preprocessed_balanced,
    y_binned_encoded_train_balanced,
)


# SHAP Explainers for the Random Forest models
rand_for_feature_eval_tuned_model_shap_explainer_transformed = shap.TreeExplainer(
    rand_for_feature_eval_tuned_model_transformed,
    X_transformed_train_preprocessed_balanced,
)

rand_for_feature_eval_tuned_model_shap_explainer_binned_labeled = shap.TreeExplainer(
    rand_for_feature_eval_tuned_model_binned_labeled,
    X_binned_labeled_train_preprocessed_balanced,
)

rand_for_feature_eval_tuned_model_shap_explainer_binned_encoded = shap.TreeExplainer(
    rand_for_feature_eval_tuned_model_binned_encoded,
    X_binned_encoded_train_preprocessed_balanced,
)


# SHAP Values for the Random Forest models
rand_for_feature_eval_tuned_model_shap_values_transformed = (
    rand_for_feature_eval_tuned_model_shap_explainer_transformed.shap_values(
        X_transformed_train_preprocessed_balanced
    )
)

rand_for_feature_eval_tuned_model_shap_values_binned_labeled = (
    rand_for_feature_eval_tuned_model_shap_explainer_binned_labeled.shap_values(
        X_binned_labeled_train_preprocessed_balanced,
        check_additivity=False,
    )
)

rand_for_feature_eval_tuned_model_shap_values_binned_encoded = (
    rand_for_feature_eval_tuned_model_shap_explainer_binned_encoded.shap_values(
        X_binned_encoded_train_preprocessed_balanced
    )
)


# Mean absolute SHAP values for each feature for the Random Forest models
sample_mean_abs_rand_for_feature_eval_tuned_model_shap_values_transformed = np.mean(
    np.abs(rand_for_feature_eval_tuned_model_shap_values_transformed),
    axis=0,
)

sample_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_labeled = np.mean(
    np.abs(rand_for_feature_eval_tuned_model_shap_values_binned_labeled),
    axis=0,
)

sample_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_encoded = np.mean(
    np.abs(rand_for_feature_eval_tuned_model_shap_values_binned_encoded),
    axis=0,
)

class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_transformed = np.mean(
    sample_mean_abs_rand_for_feature_eval_tuned_model_shap_values_transformed,
    axis=1,
)

class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_labeled = np.mean(
    sample_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_labeled,
    axis=1,
)

class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_encoded = np.mean(
    sample_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_encoded,
    axis=1,
)

class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_transformed_df = (
    pd.DataFrame(
        {
            "Feature": X_transformed_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_transformed,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)

class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_labeled_df = (
    pd.DataFrame(
        {
            "Feature": X_binned_labeled_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_labeled,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)

class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_encoded_df = (
    pd.DataFrame(
        {
            "Feature": X_binned_encoded_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_encoded,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)


# HistGradientBoosting Classifier
hgboost_feature_eval_model = HistGradientBoostingClassifier(
    class_weight={0: 1, 1: 4},
    random_state=42,
)

hgboost_feature_eval_search_space = {
    "max_iter": Integer(100, 400),
    "max_depth": Integer(3, 10),
    "learning_rate": Real(0.01, 0.3),
    "n_estimators": Integer(50, 300),
}

hgboost_feature_eval_tuned_model_transformed = bayesian_hyperparameter_tuning(
    estimator=hgboost_feature_eval_model,
    search_spaces=hgboost_feature_eval_search_space,
    X_train=X_transformed_train_preprocessed_balanced,
    y_train=y_transformed_train_balanced,
)

hgboost_feature_eval_tuned_model_binned_labeled = bayesian_hyperparameter_tuning(
    estimator=hgboost_feature_eval_model,
    search_spaces=hgboost_feature_eval_search_space,
    X_train=X_binned_labeled_train_preprocessed_balanced,
    y_train=y_binned_labeled_train_balanced,
)

hgboost_feature_eval_tuned_model_binned_encoded = bayesian_hyperparameter_tuning(
    estimator=hgboost_feature_eval_model,
    search_spaces=hgboost_feature_eval_search_space,
    X_train=X_binned_encoded_train_preprocessed_balanced,
    y_train=y_binned_encoded_train_balanced,
)

hgboost_feature_eval_tuned_model_transformed.fit(
    X_transformed_train_preprocessed_balanced,
    y_transformed_train_balanced,
)

hgboost_feature_eval_tuned_model_binned_labeled.fit(
    X_binned_labeled_train_preprocessed_balanced,
    y_binned_labeled_train_balanced,
)

hgboost_feature_eval_tuned_model_binned_encoded.fit(
    X_binned_encoded_train_preprocessed_balanced,
    y_binned_encoded_train_balanced,
)


# Sampling 100 random points to reduce the computational time of the KernelExplainer
k = 100
X_transformed_train_preprocessed_balanced_sampled = shap.sample(
    X_transformed_train_preprocessed_balanced,
    k,
)
X_binned_labeled_train_preprocessed_balanced_sampled = shap.sample(
    X_binned_labeled_train_preprocessed_balanced,
    k,
)
X_binned_encoded_train_preprocessed_balanced_sampled = shap.sample(
    X_binned_encoded_train_preprocessed_balanced,
    k,
)


# SHAP Explainers for the HistGradientBoosting models
hgboost_feature_eval_tuned_model_shap_explainer_transformed = shap.KernelExplainer(
    hgboost_feature_eval_tuned_model_transformed.predict,
    X_transformed_train_preprocessed_balanced_sampled,
)

hgboost_feature_eval_tuned_model_shap_explainer_binned_labeled = shap.KernelExplainer(
    hgboost_feature_eval_tuned_model_binned_labeled.predict,
    X_binned_labeled_train_preprocessed_balanced_sampled,
)

hgboost_feature_eval_tuned_model_shap_explainer_binned_encoded = shap.KernelExplainer(
    hgboost_feature_eval_tuned_model_binned_encoded.predict,
    X_binned_encoded_train_preprocessed_balanced_sampled,
)


# SHAP Values for the HistGradientBoosting models
hgboost_feature_eval_tuned_model_shap_values_transformed = (
    hgboost_feature_eval_tuned_model_shap_explainer_transformed.shap_values(
        X_transformed_train_preprocessed_balanced_sampled,
    )
)

hgboost_feature_eval_tuned_model_shap_values_binned_labeled = (
    hgboost_feature_eval_tuned_model_shap_explainer_binned_labeled.shap_values(
        X_binned_labeled_train_preprocessed_balanced_sampled
    )
)

hgboost_feature_eval_tuned_model_shap_values_binned_encoded = (
    hgboost_feature_eval_tuned_model_shap_explainer_binned_encoded.shap_values(
        X_binned_encoded_train_preprocessed_balanced_sampled
    )
)


# Mean absolute SHAP values for each feature for the HistGradientBoosting models
sample_mean_abs_hgboost_feature_eval_tuned_model_shap_values_transformed = np.mean(
    np.abs(hgboost_feature_eval_tuned_model_shap_values_transformed),
    axis=0,
)

sample_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_labeled = np.mean(
    np.abs(hgboost_feature_eval_tuned_model_shap_values_binned_labeled),
    axis=0,
)

sample_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_encoded = np.mean(
    np.abs(hgboost_feature_eval_tuned_model_shap_values_binned_encoded),
    axis=0,
)

class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_transformed_df = (
    pd.DataFrame(
        {
            "Feature": X_transformed_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": sample_mean_abs_hgboost_feature_eval_tuned_model_shap_values_transformed,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)

class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_labeled_df = (
    pd.DataFrame(
        {
            "Feature": X_binned_labeled_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": sample_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_labeled,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)

class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_encoded_df = (
    pd.DataFrame(
        {
            "Feature": X_binned_encoded_train_preprocessed_balanced.columns,
            "Mean Absolute SHAP Value": sample_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_encoded,
        }
    )
    .sort_values(by="Mean Absolute SHAP Value", ascending=False)
    .reset_index(drop=True)
)


# Prune features below the 20th percentile for each model
percentile_threshold = 20


# Calculate the threshold value based on percentile
# Logistic Regression Classifier
threshold_log_reg_transformed_df = np.percentile(
    mean_abs_log_reg_feature_eval_tuned_model_shap_values_transformed_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)

threshold_log_reg_binned_labeled_df = np.percentile(
    mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_labeled_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)

threshold_log_reg_binned_encoded_df = np.percentile(
    mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_encoded_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)

# Random Forest Classifier
threshold_rand_for_transformed_df = np.percentile(
    class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_transformed_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)

threshold_rand_for_binned_labeled_df = np.percentile(
    class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_labeled_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)

threshold_rand_for_binned_encoded_df = np.percentile(
    class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_encoded_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)

# HistGradientBoosting Classifier
threshold_xgboost_transformed_df = np.percentile(
    class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_transformed_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)

threshold_xgboost_binned_labeled_df = np.percentile(
    class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_labeled_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)

threshold_xgboost_binned_encoded_df = np.percentile(
    class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_encoded_df[
        "Mean Absolute SHAP Value"
    ],
    percentile_threshold,
)


# Prune features
# Logistic Regression Classifier
pruned_log_reg_transformed_df = (
    mean_abs_log_reg_feature_eval_tuned_model_shap_values_transformed_df[
        mean_abs_log_reg_feature_eval_tuned_model_shap_values_transformed_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_log_reg_transformed_df
    ]
)

pruned_log_reg_binned_labeled_df = (
    mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_labeled_df[
        mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_labeled_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_log_reg_binned_labeled_df
    ]
)

pruned_log_reg_binned_encoded_df = (
    mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_encoded_df[
        mean_abs_log_reg_feature_eval_tuned_model_shap_values_binned_encoded_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_log_reg_binned_encoded_df
    ]
)

# Random Forest Classifier
pruned_rand_for_transformed_df = (
    class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_transformed_df[
        class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_transformed_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_rand_for_transformed_df
    ]
)

pruned_rand_for_binned_labeled_df = (
    class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_labeled_df[
        class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_labeled_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_rand_for_binned_labeled_df
    ]
)

pruned_rand_for_binned_encoded_df = (
    class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_encoded_df[
        class_mean_abs_rand_for_feature_eval_tuned_model_shap_values_binned_encoded_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_rand_for_binned_encoded_df
    ]
)

# HistGradientBoosting Classifier
pruned_xgboost_transformed_df = (
    class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_transformed_df[
        class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_transformed_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_xgboost_transformed_df
    ]
)

pruned_xgboost_binned_labeled_df = (
    class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_labeled_df[
        class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_labeled_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_xgboost_binned_labeled_df
    ]
)

pruned_xgboost_binned_encoded_df = (
    class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_encoded_df[
        class_mean_abs_hgboost_feature_eval_tuned_model_shap_values_binned_encoded_df[
            "Mean Absolute SHAP Value"
        ]
        >= threshold_xgboost_binned_encoded_df
    ]
)


# Define the intersections of the remaining features
intersection_features_transformed = set(
    pruned_log_reg_transformed_df["Feature"]
).intersection(
    set(pruned_rand_for_transformed_df["Feature"]).intersection(
        set(pruned_xgboost_transformed_df["Feature"])
    )
)

intersection_features_binned_labeled = set(
    pruned_log_reg_binned_labeled_df["Feature"]
).intersection(
    set(pruned_rand_for_binned_labeled_df["Feature"]).intersection(
        set(pruned_xgboost_binned_labeled_df["Feature"])
    )
)

intersection_features_binned_encoded = set(
    pruned_log_reg_binned_encoded_df["Feature"]
).intersection(
    set(pruned_rand_for_binned_encoded_df["Feature"]).intersection(
        set(pruned_xgboost_binned_encoded_df["Feature"])
    )
)


#
X_transformed_train_preprocessed_balanced_optimized = (
    X_transformed_train_preprocessed_balanced[list(intersection_features_transformed)]
)

X_binned_labeled_train_preprocessed_balanced_optimized = (
    X_binned_labeled_train_preprocessed_balanced[
        list(intersection_features_binned_labeled)
    ]
)

X_binned_encoded_train_preprocessed_balanced_optimized = (
    X_binned_encoded_train_preprocessed_balanced[
        list(intersection_features_binned_encoded)
    ]
)


# VIF Determination for _transformed
vif_X_transformed_train_preprocessed_balanced_optimized = pd.DataFrame()
vif_X_transformed_train_preprocessed_balanced_optimized["Feature"] = (
    X_transformed_train_preprocessed_balanced_optimized.columns
)
vif_X_transformed_train_preprocessed_balanced_optimized["VIF"] = [
    variance_inflation_factor(
        X_transformed_train_preprocessed_balanced_optimized.values,
        i,
    )
    for i in range(len(X_transformed_train_preprocessed_balanced_optimized.columns))
]


# VIF Determination for _binned_labeled
vif_X_binned_labeled_train_preprocessed_balanced_optimized = pd.DataFrame()
vif_X_binned_labeled_train_preprocessed_balanced_optimized["Feature"] = (
    X_binned_labeled_train_preprocessed_balanced_optimized.columns
)
vif_X_binned_labeled_train_preprocessed_balanced_optimized["VIF"] = [
    variance_inflation_factor(
        X_binned_labeled_train_preprocessed_balanced_optimized.values,
        i,
    )
    for i in range(len(X_binned_labeled_train_preprocessed_balanced_optimized.columns))
]


# VIF Determination for _binned_encoded
vif_X_binned_encoded_train_preprocessed_balanced_optimized = pd.DataFrame()
vif_X_binned_encoded_train_preprocessed_balanced_optimized["Feature"] = (
    X_binned_encoded_train_preprocessed_balanced_optimized.columns
)
vif_X_binned_encoded_train_preprocessed_balanced_optimized["VIF"] = [
    variance_inflation_factor(
        X_binned_encoded_train_preprocessed_balanced_optimized.values,
        i,
    )
    for i in range(len(X_binned_encoded_train_preprocessed_balanced_optimized.columns))
]


# Prune features based on the VIF threshold
vif_threshold = 10


# VIF Pruning for _transformed
vif_pruned_X_transformed_train_preprocessed_balanced_optimized = (
    vif_X_transformed_train_preprocessed_balanced_optimized[
        vif_X_transformed_train_preprocessed_balanced_optimized["VIF"] <= 10
    ].reset_index(drop=True)
)


# VIF Pruning for _binned_labeled
vif_pruned_X_binned_labeled_train_preprocessed_balanced_optimized = (
    vif_X_binned_labeled_train_preprocessed_balanced_optimized[
        vif_X_binned_labeled_train_preprocessed_balanced_optimized["VIF"] <= 10
    ].reset_index(drop=True)
)


# VIF Pruning for _binned_encoded
vif_pruned_X_binned_encoded_train_preprocessed_balanced_optimized = (
    vif_X_binned_encoded_train_preprocessed_balanced_optimized[
        vif_X_binned_encoded_train_preprocessed_balanced_optimized["VIF"] <= 10
    ].reset_index(drop=True)
)


X_optimized_transformed = X_transformed_train_preprocessed_balanced[
    vif_pruned_X_transformed_train_preprocessed_balanced_optimized["Feature"]
]
X_optimized_new_binned_labeled = X_binned_labeled_train_preprocessed_balanced[
    vif_pruned_X_binned_labeled_train_preprocessed_balanced_optimized["Feature"]
]
X_optimized_new_binned_encoded = X_binned_encoded_train_preprocessed_balanced[
    vif_pruned_X_binned_encoded_train_preprocessed_balanced_optimized["Feature"]
]


X_new_transformed_test_preprocessed = X_transformed_test_preprocessed[
    vif_pruned_X_transformed_train_preprocessed_balanced_optimized["Feature"]
]
X_new_binned_labeled_test_preprocessed = X_binned_labeled_test_preprocessed[
    vif_pruned_X_binned_labeled_train_preprocessed_balanced_optimized["Feature"]
]
X_new_binned_encoded_test_preprocessed = X_binned_encoded_test_preprocessed[
    vif_pruned_X_binned_encoded_train_preprocessed_balanced_optimized["Feature"]
]
