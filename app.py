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
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Personality Analysis Classifier",
    page_icon="📊",
    layout="wide"
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
    "Random Forest (Ensemble)": "model/random_forest.pkl"
}




# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("1. Upload Test Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

st.sidebar.header("2. Select Model")

selected_model = st.sidebar.selectbox(
    "Choose a Classification Model",
    list(model_paths.keys())
)

st.sidebar.markdown("---")
st.sidebar.write("### Expected CSV Format")
st.sidebar.write(
    "Your test CSV should contain the same predictor columns used during "
    "model training and a `Response` column for evaluation."
)


# ---------------------------------------------------------
# MODEL PERFORMANCE SUMMARY
# ---------------------------------------------------------
st.write("## 📈 Model Performance Summary")

results_df = pd.DataFrame(model_results).T
results_df.index.name = "ML Model Name"

st.dataframe(
    results_df.style.format("{:.4f}"),
    use_container_width=True
)



observation_df = pd.DataFrame(
    [
        {"ML Model Name": model, "Observation about model performance": text}
        for model, text in observations.items()
    ]
)

st.dataframe(
    observation_df,
    use_container_width=True,
    hide_index=True
)




# ---------------------------------------------------------
# TEST DATA EVALUATION
# ---------------------------------------------------------
if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.write("## 📊 Uploaded Test Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        st.write(
            f"**Dataset Shape:** {data.shape[0]} rows × {data.shape[1]} columns"
        )

        # -------------------------------------------------
        # Validate Target Column
        # -------------------------------------------------
        if "Response" not in data.columns:

            st.error(
                "❌ The uploaded CSV must contain a `Response` column "
                "for model evaluation."
            )

            st.stop()

        # Separate target and predictors
        X_test = data.drop(columns=["Response"])
        y_test = data["Response"]

        # Make sure target is numeric
        try:
            y_test = pd.to_numeric(y_test)
        except Exception:
            st.error("The `Response` column must contain binary values 0 and 1.")
            st.stop()

        unique_targets = sorted(y_test.dropna().unique().tolist())

        if not set(unique_targets).issubset({0, 1}):
            st.error(
                f"`Response` must contain only 0 and 1. "
                f"Found values: {unique_targets}"
            )
            st.stop()

        # -------------------------------------------------
        # Load Model
        # -------------------------------------------------
        st.write(f"## 🤖 Evaluation: {selected_model}")

        try:
            model = joblib.load(model_paths[selected_model])

        except FileNotFoundError:
            st.error(
                f"Model file not found: `{model_paths[selected_model]}`. "
                "Please make sure the required .pkl files are inside the `model/` folder."
            )
            st.stop()

        # -------------------------------------------------
        # Load Scaler
        # -------------------------------------------------
        scaler = None

        try:
            scaler = joblib.load("model/scaler.pkl")
        except FileNotFoundError:
            st.warning(
                "⚠️ `model/scaler.pkl` was not found. "
                "The application will attempt to use the uploaded features directly."
            )

        # -------------------------------------------------
        # Feature Alignment
        # -------------------------------------------------
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)

            missing_features = [
                col for col in expected_features
                if col not in X_test.columns
            ]

            extra_features = [
                col for col in X_test.columns
                if col not in expected_features
            ]

            if missing_features:
                st.error(
                    "The uploaded test data is missing the following model features: "
                    + ", ".join(missing_features)
                )
                st.stop()

            X_test = X_test[expected_features]

            if extra_features:
                st.warning(
                    "Extra columns were ignored: "
                    + ", ".join(extra_features)
                )

        # -------------------------------------------------
        # Handle Missing Values
        # -------------------------------------------------
        X_test = X_test.copy()

        numeric_columns = X_test.select_dtypes(
            include=["number"]
        ).columns

        X_test[numeric_columns] = X_test[numeric_columns].fillna(
            X_test[numeric_columns].median()
        )

        # -------------------------------------------------
        # Transform / Scale
        # -------------------------------------------------
        if scaler is not None:
            try:
                X_processed = scaler.transform(X_test)
            except Exception as scaler_error:
                st.warning(
                    f"Scaler could not transform the uploaded data: {scaler_error}. "
                    "Using the processed features directly."
                )
                X_processed = X_test
        else:
            X_processed = X_test

        # -------------------------------------------------
        # Predictions
        # -------------------------------------------------
        y_pred = model.predict(X_processed)

        # Probability / AUC
        y_prob = None

        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_processed)[:, 1]
            except Exception:
                y_prob = None

        if y_prob is not None:
            try:
                auc = roc_auc_score(y_test, y_prob)
            except ValueError:
                auc = None
        else:
            try:
                auc = roc_auc_score(y_test, y_pred)
            except ValueError:
                auc = None

        # -------------------------------------------------
        # Evaluation Metrics
        # -------------------------------------------------
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
        mcc = matthews_corrcoef(
            y_test, y_pred
        )

        st.write("### 📌 Evaluation Metrics")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        c2.metric(
            "AUC Score",
            f"{auc:.4f}" if auc is not None else "N/A"
        )

        c3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        c4, c5, c6 = st.columns(3)

        c4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        c5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        c6.metric(
            "MCC Score",
            f"{mcc:.4f}"
        )

        # -------------------------------------------------
        # Prediction Distribution
        # -------------------------------------------------
        st.write("### 🎯 Prediction Summary")

        positive_predictions = int((y_pred == 1).sum())
        negative_predictions = int((y_pred == 0).sum())

        p1, p2 = st.columns(2)

        p1.metric(
            "Predicted Acceptances",
            positive_predictions
        )

        p2.metric(
            "Predicted Rejections",
            negative_predictions
        )

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------
        st.write("### 🧩 Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=[
                "Rejected (0)",
                "Accepted (1)"
            ],
            yticklabels=[
                "Rejected (0)",
                "Accepted (1)"
            ]
        )

        ax.set_ylabel("Actual Response")
        ax.set_xlabel("Predicted Response")
        ax.set_title(f"{selected_model} - Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------------------------
        # Prediction Results
        # -------------------------------------------------
        st.write("### 🔍 Prediction Results")

        results_output = data.copy()
        results_output["Predicted_Response"] = y_pred

        if y_prob is not None:
            results_output["Acceptance_Probability"] = y_prob

        st.dataframe(
            results_output.head(100),
            use_container_width=True
        )

        # -------------------------------------------------
        # Download Predictions
        # -------------------------------------------------
        csv_data = results_output.to_csv(index=False)

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_data,
            file_name="customer_personality_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"❌ An error occurred while processing the uploaded file: {e}"
        )

else:

    st.info(
        "👈 Please upload the `test_data.csv` file from the sidebar "
        "to evaluate the selected model."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")

st.caption(
    "Kaggle Customer Personality Analysis Classification | "
    "Machine Learning Assignment 2"
)
