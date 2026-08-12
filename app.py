## ChatGpt version:

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Personality Analysis Classifier",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Customer Personality Analysis Classification")
st.markdown("### ML Assignment 2")
st.write(
    "This application predicts whether a customer will accept a company's "
    "marketing campaign offer using multiple machine learning classification models."
)

st.info(
    "**Target Variable:** `Response` — 1 = Customer accepted the campaign offer, "
    "0 = Customer did not accept the campaign offer."
)


# ---------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------
model_paths = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbor (KNN)": "model/knn.pkl",
    "Gaussian Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest.pkl",
}


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def load_model(model_name):
    """Load a saved model from the model folder."""
    return joblib.load(model_paths[model_name])


def get_expected_features(model, scaler):
    """
    Determine the feature order used during training.

    Priority:
    1. model.feature_names_in_
    2. scaler.feature_names_in_
    3. model/feature_columns.pkl
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)

    try:
        feature_columns = joblib.load("model/feature_columns.pkl")
        return list(feature_columns)
    except (FileNotFoundError, Exception):
        return None


def prepare_features(data, model):
    """
    Prepare uploaded predictor data using the saved training scaler.
    No dataset values or evaluation results are hardcoded.
    """
    X = data.drop(columns=["Response"]).copy()

    try:
        scaler = joblib.load("model/scaler.pkl")
    except FileNotFoundError:
        scaler = None

    expected_features = get_expected_features(model, scaler)

    if expected_features is not None:
        missing_features = [
            col for col in expected_features if col not in X.columns
        ]

        if missing_features:
            raise ValueError(
                "The uploaded CSV is missing these required model features: "
                + ", ".join(missing_features)
            )

        extra_features = [col for col in X.columns if col not in expected_features]
        X = X[expected_features]

        if extra_features:
            st.warning(
                "Extra columns were ignored: " + ", ".join(extra_features)
            )

    # Fill missing numeric values using the uploaded test data.
    numeric_columns = X.select_dtypes(include=["number"]).columns
    X[numeric_columns] = X[numeric_columns].fillna(
        X[numeric_columns].median()
    )

    if scaler is not None:
        X_processed = scaler.transform(X)
    else:
        X_processed = X

    return X_processed


def evaluate_model(model_name, data):
    """
    Evaluate one saved model against the uploaded CSV.
    All metrics are calculated from the uploaded file at runtime.
    """
    model = load_model(model_name)

    X_processed = prepare_features(data, model)
    y_test = pd.to_numeric(data["Response"])

    y_pred = model.predict(X_processed)

    y_prob = None
    auc = None

    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_processed)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except (ValueError, IndexError):
            auc = None
    elif hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X_processed)
            auc = roc_auc_score(y_test, y_score)
        except ValueError:
            auc = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": auc,
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "MCC Score": matthews_corrcoef(
            y_test, y_pred
        ),
    }

    return metrics, y_pred, y_prob


def generate_observation(model_name, metrics_df):
    """
    Generate an observation from the metrics calculated from the
    uploaded CSV. No performance numbers are hardcoded.
    """
    row = metrics_df.loc[model_name]

    accuracy = row["Accuracy"]
    auc = row["AUC Score"]
    precision = row["Precision"]
    recall = row["Recall"]
    f1 = row["F1 Score"]
    mcc = row["MCC Score"]

    observations = []

    # Relative performance
    if accuracy == metrics_df["Accuracy"].max():
        observations.append("achieved the highest Accuracy among the evaluated models")
    elif accuracy >= metrics_df["Accuracy"].median():
        observations.append("achieved competitive Accuracy")

    if pd.notna(auc):
        if auc == metrics_df["AUC Score"].max():
            observations.append("achieved the highest AUC, indicating the strongest class discrimination")
        elif auc >= metrics_df["AUC Score"].median():
            observations.append("showed competitive class discrimination")

    if precision == metrics_df["Precision"].max():
        observations.append("had the highest Precision, indicating relatively fewer false-positive predictions")

    if recall == metrics_df["Recall"].max():
        observations.append("had the highest Recall, indicating that it identified the largest share of positive cases")

    if f1 == metrics_df["F1 Score"].max():
        observations.append("achieved the highest F1 Score, giving the strongest balance between Precision and Recall")

    if mcc == metrics_df["MCC Score"].max():
        observations.append("achieved the highest MCC, indicating the strongest overall balanced correlation between predictions and actual classes")

    # Model-specific interpretation based on its observed metrics
    if precision > recall + 0.10:
        observations.append(
            "its higher Precision than Recall suggests a more conservative positive-class prediction strategy"
        )
    elif recall > precision + 0.10:
        observations.append(
            "its higher Recall than Precision suggests a more aggressive positive-class prediction strategy"
        )

    if not observations:
        observations.append("showed a balanced performance across the evaluated metrics")

    return (
        f"{model_name} {observations[0]}."
        + (" " + ". ".join(observations[1:]) + "." if len(observations) > 1 else "")
    )


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("1. Upload Test Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
)

st.sidebar.header("2. Select Model")

selected_model = st.sidebar.selectbox(
    "Choose a Classification Model",
    list(model_paths.keys()),
)

st.sidebar.markdown("---")
st.sidebar.write("### Expected CSV Format")
st.sidebar.write(
    "Upload a CSV containing the same predictor columns used during model "
    "training and a `Response` column containing 0 and 1."
)


# ---------------------------------------------------------
# WAIT FOR USER TO UPLOAD CSV
# ---------------------------------------------------------
if uploaded_file is None:
    st.info(
        "👈 Please upload the `test_data.csv` file from the sidebar. "
        "Model performance, observations, and predictions will be generated "
        "only after the CSV is uploaded."
    )

else:
    try:
        data = pd.read_csv(uploaded_file)

        st.write("## 📊 Uploaded Test Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        st.write(
            f"**Dataset Shape:** {data.shape[0]} rows × {data.shape[1]} columns"
        )

        # -------------------------------------------------
        # VALIDATE TARGET
        # -------------------------------------------------
        if "Response" not in data.columns:
            st.error(
                "❌ The uploaded CSV must contain a `Response` column "
                "for model evaluation."
            )
            st.stop()

        y_test = pd.to_numeric(data["Response"], errors="coerce")

        if y_test.isna().any():
            st.error("The `Response` column must contain only binary values 0 and 1.")
            st.stop()

        unique_targets = sorted(y_test.unique().tolist())

        if not set(unique_targets).issubset({0, 1}):
            st.error(
                f"`Response` must contain only 0 and 1. "
                f"Found values: {unique_targets}"
            )
            st.stop()

        # -------------------------------------------------
        # GENERATE PERFORMANCE ONLY AFTER UPLOAD
        # -------------------------------------------------
        st.write("## 📈 Model Performance Summary")

        model_results = {}
        model_predictions = {}
        model_probabilities = {}

        progress = st.progress(0)

        for i, model_name in enumerate(model_paths.keys(), start=1):
            try:
                metrics, predictions, probabilities = evaluate_model(
                    model_name, data
                )

                model_results[model_name] = metrics
                model_predictions[model_name] = predictions
                model_probabilities[model_name] = probabilities

            except FileNotFoundError:
                st.error(
                    f"Model file for **{model_name}** was not found. "
                    f"Expected: `{model_paths[model_name]}`"
                )
            except Exception as model_error:
                st.error(
                    f"Could not evaluate **{model_name}**: {model_error}"
                )

            progress.progress(i / len(model_paths))

        progress.empty()

        if not model_results:
            st.error(
                "No models could be evaluated. Please check that the required "
                ".pkl files are present in the `model/` folder."
            )
            st.stop()

        # Convert runtime-generated results into a DataFrame.
        results_df = pd.DataFrame(model_results).T
        results_df.index.name = "ML Model Name"

        st.dataframe(
            results_df.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "AUC Score": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                    "F1 Score": "{:.4f}",
                    "MCC Score": "{:.4f}",
                },
                na_rep="N/A",
            ),
            use_container_width=True,
        )

        # -------------------------------------------------
        # DYNAMIC OBSERVATIONS
        # -------------------------------------------------
        st.write("## 📝 Observations")

        observation_df = pd.DataFrame(
            [
                {
                    "ML Model Name": model_name,
                    "Observation about model performance": generate_observation(
                        model_name, results_df
                    ),
                }
                for model_name in model_results.keys()
            ]
        )

        st.dataframe(
            observation_df,
            use_container_width=True,
            hide_index=True,
        )

        # -------------------------------------------------
        # OVERALL WINNER
        # -------------------------------------------------
        winner = results_df["F1 Score"].idxmax()

        st.success(
            f"🏆 **Overall Winner based on the highest F1 Score: {winner}**"
        )

        # -------------------------------------------------
        # SELECTED MODEL EVALUATION
        # -------------------------------------------------
        st.write(f"## 🤖 Evaluation: {selected_model}")

        if selected_model not in model_predictions:
            st.error(
                f"No prediction results are available for {selected_model}."
            )
            st.stop()

        y_pred = model_predictions[selected_model]
        y_prob = model_probabilities[selected_model]

        selected_metrics = results_df.loc[selected_model]

        # -------------------------------------------------
        # EVALUATION METRICS
        # -------------------------------------------------
        st.write("### 📌 Evaluation Metrics")

        c1, c2, c3 = st.columns(3)

        c1.metric("Accuracy", f"{selected_metrics['Accuracy']:.4f}")
        c2.metric(
            "AUC Score",
            f"{selected_metrics['AUC Score']:.4f}"
            if pd.notna(selected_metrics["AUC Score"])
            else "N/A",
        )
        c3.metric("Precision", f"{selected_metrics['Precision']:.4f}")

        c4, c5, c6 = st.columns(3)

        c4.metric("Recall", f"{selected_metrics['Recall']:.4f}")
        c5.metric("F1 Score", f"{selected_metrics['F1 Score']:.4f}")
        c6.metric("MCC Score", f"{selected_metrics['MCC Score']:.4f}")

        # -------------------------------------------------
        # PREDICTION DISTRIBUTION
        # -------------------------------------------------
        st.write("### 🎯 Prediction Summary")

        positive_predictions = int((y_pred == 1).sum())
        negative_predictions = int((y_pred == 0).sum())

        p1, p2 = st.columns(2)

        p1.metric("Predicted Acceptances", positive_predictions)
        p2.metric("Predicted Rejections", negative_predictions)

        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------
        st.write("### 🧩 Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Rejected (0)", "Accepted (1)"],
            yticklabels=["Rejected (0)", "Accepted (1)"],
        )

        ax.set_ylabel("Actual Response")
        ax.set_xlabel("Predicted Response")
        ax.set_title(f"{selected_model} - Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------------------------
        # PREDICTION RESULTS
        # -------------------------------------------------
        st.write("### 🔍 Prediction Results")

        results_output = data.copy()
        results_output["Predicted_Response"] = y_pred

        if y_prob is not None:
            results_output["Acceptance_Probability"] = y_prob

        st.dataframe(
            results_output.head(100),
            use_container_width=True,
        )

        # -------------------------------------------------
        # DOWNLOAD PREDICTIONS
        # -------------------------------------------------
        csv_data = results_output.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="customer_personality_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(
            f"❌ An error occurred while processing the uploaded file: {e}"
        )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Kaggle Customer Personality Analysis Classification | "
    "Machine Learning Assignment 2"
)
