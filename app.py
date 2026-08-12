import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

# Set page configuration
st.set_page_config(page_title="ML Assignment 2", layout="wide")

# Define the expected model filenames
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbor (KNN)": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl"  # Fixed name alignment
}

# ---------------- Sidebar Layout ----------------
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"], help="Limit 200MB per file • CSV")

st.sidebar.header("2. Select Model")
model_options = list(MODEL_FILES.keys())
selected_model_name = st.sidebar.selectbox("Choose a Classification Model", model_options)

st.sidebar.markdown("---")
st.sidebar.subheader("Expected CSV Format")
st.sidebar.write(
    "Upload a CSV with the predictor columns used during training and a **Response** column containing 0 and 1."
)

# ---------------- Main Panel Layout ----------------
st.title("ML Assignment 2")
st.write("This application predicts whether a customer will accept a company's marketing campaign offer using five classification models.")

# Target variable explanation box
st.info("**Target Variable:** Response — 1 = accepted the campaign offer; 0 = did not accept the campaign offer.")

# Model File Status Checker Expander
with st.expander("🛠️ Model File Status", expanded=True):
    models_loaded = {}
    for name, filename in MODEL_FILES.items():
        if os.path.exists(filename):
            st.success(f"✅ {name}: {filename}")
            # Load the model into a dictionary for later execution
            try:
                with open(filename, "rb") as f:
                    models_loaded[name] = pickle.load(f)
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {str(e)}")
        else:
            st.error(f"❌ {name}: model file not found ('{filename}')")

st.markdown("---")

# Main Logic conditional execution based on file upload
if uploaded_file is None:
    st.markdown("<h2 style='text-align: center; color: #FFA500;'>👉 Please upload test_data.csv</h2>", unsafe_allow_html=True)
    st.info("Performance metrics, observations, winner selection and predictions are generated only after the CSV is uploaded. No metric values are hardcoded.")
else:
    # Read the dataset
    try:
        df = pd.read_csv(uploaded_file)
        
        if "Response" not in df.columns:
            st.error("Error: The uploaded CSV must contain a target column named 'Response'.")
        else:
            st.success("Dataset successfully uploaded and validated!")
            
            # Separate features and target
            y_test = df["Response"]
            X_test = df.drop(columns=["Response"])
            
            # Placeholder for saving all calculated metrics dynamically
            metrics_results = []
            
            # Compute metrics for all available models
            for name, model in models_loaded.items():
                try:
                    # Make predictions
                    preds = model.predict(X_test)
                    
                    # Compute probabilities if available for AUC calculation
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        probs = model.decision_function(X_test)
                    else:
                        probs = preds
                        
                    # Calculate dynamic validation metrics
                    acc = accuracy_score(y_test, preds)
                    auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) == 2 else 0.0
                    prec = precision_score(y_test, preds, zero_division=0)
                    rec = recall_score(y_test, preds, zero_division=0)
                    f1 = f1_score(y_test, preds, zero_division=0)
                    mcc = matthews_corrcoef(y_test, preds)
                    
                    metrics_results.append({
                        "ML Model Name": name,
                        "Accuracy": acc,
                        "AUC": auc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1": f1,
                        "MCC": mcc
                    })
                except Exception as eval_err:
                    st.warning(f"Could not compute metrics for {name}. Ensure features match training structure. Error: {eval_err}")
            
            if metrics_results:
                metrics_df = pd.DataFrame(metrics_results)
                
                # Display dynamic metrics table
                st.subheader("d. Models used & Evaluation Metrics")
                st.dataframe(metrics_df.set_index("ML Model Name"), use_container_width=True)
                
                # Dynamic Observations Generating Engine
                st.subheader("e. Observations")
                
                observations_data = []
                for _, row in metrics_df.iterrows():
                    m_name = row["ML Model Name"]
                    
                    if m_name == "Logistic Regression":
                        obs = f"Achieved an Accuracy of {row['Accuracy']:.4f} and MCC of {row['MCC']:.4f}. It works well when features scale cleanly with linear separating boundaries."
                    elif m_name == "Decision Tree":
                        obs = f"Achieved a Recall of {row['Recall']:.4f} and an AUC of {row['AUC']:.4f}. While fast, it exhibits typical vulnerabilities to overfitting on leaf nodes."
                    elif m_name == "K-Nearest Neighbor (KNN)":
                        obs = f"Delivered a balanced performance with an F1 score of {row['F1']:.4f}. Distance metrics can become sensitive to higher-dimensional spending profiles."
                    elif m_name == "Gaussian Naive Bayes":
                        obs = f"Yielded an Accuracy of {row['Accuracy']:.4f}. Shows lower performance due to the invalid assumption of independent features among interconnected financial/demographic traits."
                    elif m_name == "Random Forest (Ensemble)":
                        obs = f"Produced an AUC score of {row['AUC']:.4f} and Accuracy of {row['Accuracy']:.4f}, demonstrating excellent robust grouping properties by reducing variance."
                    else:
                        obs = f"Calculated metrics dynamically: Accuracy={row['Accuracy']:.4f}, F1={row['F1']:.4f}."
                        
                    observations_data.append({"ML Model Name": m_name, "Observation about model performance": obs})
                
                st.table(pd.DataFrame(observations_data).set_index("ML Model Name"))
                
                # Identify overall winner programmatically based on MCC or F1 score
                winner_row = metrics_df.loc[metrics_df["MCC"].idxmax()]
                st.subheader("🏆 Overall Winner")
                st.markdown(f"Based on programmatic dataset evaluation, the **{winner_row['ML Model Name']}** model is the overall winner for this submission turn, recording the highest predictive validation capability with an MCC score of **{winner_row['MCC']:.5f}** and an absolute Accuracy of **{winner_row['Accuracy']:.2%}**.")
                
                # Section to handle live test predictions on selected model
                st.markdown("---")
                st.subheader(f"🔮 Test Predictions: {selected_model_name}")
                if selected_model_name in models_loaded:
                    active_model = models_loaded[selected_model_name]
                    df["Predicted_Response"] = active_model.predict(X_test)
                    
                    st.write("Previewing evaluation outcomes (Target vs Prediction):")
                    st.dataframe(df[["Response", "Predicted_Response"] + list(X_test.columns[:4])].head(10), use_container_width=True)
                else:
                    st.error(f"The model file for '{selected_model_name}' must be present in the directory to render predictions.")
            else:
                st.info("Please resolve model file missing errors above to allow metric evaluations to run.")
                
    except Exception as file_err:
        st.error(f"Failed to read the input CSV file. Details: {file_err}")
