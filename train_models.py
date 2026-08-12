"""
train_models.py
Kaggle Customer Personality Analysis - Classification

Purpose:
1. Load the Kaggle Customer Personality Analysis dataset.
2. Preprocess the customer data.
3. Create an 80/20 train-test split.
4. Save the processed test set as BOTH:
      - test.csv
      - test_data.csv
   (test_data.csv is retained for compatibility with the Streamlit app.)
5. Train five required classification models.
6. Save the trained models and scaler under model/.
7. Print Accuracy, AUC, Precision, Recall, F1 and MCC for the README.

Expected raw dataset file:
    marketing_campaign.csv

You can also name the raw dataset:
    data.csv
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
possible_files = [
    "marketing_campaign.csv",
    "data.csv",
    "customer_personality_analysis.csv",
]

data_file = next((f for f in possible_files if os.path.exists(f)), None)

if data_file is None:
    raise FileNotFoundError(
        "Dataset not found. Place the Kaggle Customer Personality Analysis "
        "CSV in the same folder as this script and name it "
        "'marketing_campaign.csv', 'data.csv', or "
        "'customer_personality_analysis.csv'."
    )

print(f"Loading dataset: {data_file}")

# Kaggle Customer Personality Analysis is commonly tab-separated.
try:
    df = pd.read_csv(data_file, sep="\t")
    if len(df.columns) <= 2:
        df = pd.read_csv(data_file)
except Exception:
    df = pd.read_csv(data_file)

df.columns = df.columns.str.strip()

print(f"Original dataset shape: {df.shape}")


# ---------------------------------------------------------
# 2. BASIC CLEANING
# ---------------------------------------------------------
# Remove completely empty columns.
df = df.dropna(axis=1, how="all")

if "Response" not in df.columns:
    raise ValueError(
        "Target column 'Response' was not found in the dataset."
    )

# Convert known numeric columns safely.
numeric_candidates = [
    "Year_Birth",
    "Income",
    "Kidhome",
    "Teenhome",
    "Recency",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
    "AcceptedCmp1",
    "AcceptedCmp2",
    "Complain",
    "Z_CostContact",
    "Z_Revenue",
    "Response",
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Target must be binary.
df["Response"] = pd.to_numeric(df["Response"], errors="coerce")
df = df[df["Response"].isin([0, 1])].copy()
df["Response"] = df["Response"].astype(int)


# ---------------------------------------------------------
# 3. REMOVE IDENTIFIERS / NON-PREDICTIVE DATE FIELD
# ---------------------------------------------------------
# ID is an identifier and should not be used as a predictive feature.
# Dt_Customer is a date field. We convert it to customer tenure-related
# components instead of feeding the raw string to the models.

if "Dt_Customer" in df.columns:
    dt = pd.to_datetime(
        df["Dt_Customer"],
        errors="coerce",
        dayfirst=True
    )

    # Use year/month as numeric customer-registration information.
    df["Customer_Join_Year"] = dt.dt.year
    df["Customer_Join_Month"] = dt.dt.month

    df = df.drop(columns=["Dt_Customer"])

if "ID" in df.columns:
    df = df.drop(columns=["ID"])


# ---------------------------------------------------------
# 4. OPTIONAL FEATURE ENGINEERING
# ---------------------------------------------------------
# These aggregate features summarize overall customer behaviour.

spend_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

available_spend = [c for c in spend_columns if c in df.columns]

if available_spend:
    df["Total_Spending"] = df[available_spend].sum(axis=1)

purchase_columns = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]

available_purchases = [c for c in purchase_columns if c in df.columns]

if available_purchases:
    df["Total_Purchases"] = df[available_purchases].sum(axis=1)

children_columns = [
    c for c in ["Kidhome", "Teenhome"] if c in df.columns
]

if children_columns:
    df["Total_Children_Home"] = df[children_columns].sum(axis=1)


# ---------------------------------------------------------
# 5. SEPARATE FEATURES AND TARGET
# ---------------------------------------------------------
X = df.drop(columns=["Response"]).copy()
y = df["Response"].copy()

# Replace infinite values.
X = X.replace([np.inf, -np.inf], np.nan)

# Identify categorical and numeric columns.
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

# Median imputation for numeric variables.
for col in numeric_columns:
    X[col] = X[col].fillna(X[col].median())

# Mode imputation for categorical variables.
for col in categorical_columns:
    mode = X[col].mode(dropna=True)
    fill_value = mode.iloc[0] if not mode.empty else "Unknown"
    X[col] = X[col].fillna(fill_value)

# One-hot encode categorical variables.
if categorical_columns:
    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False,
        dtype=int
    )

# Make sure everything is numeric.
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

print(f"Processed feature count: {X.shape[1]}")
print(f"Class distribution:\n{y.value_counts().sort_index()}")


# ---------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(
    f"\nTraining set: {X_train.shape[0]} rows"
    f"\nTest set:     {X_test.shape[0]} rows"
)


# ---------------------------------------------------------
# 7. SCALE FEATURES
# ---------------------------------------------------------
# The scaler is fitted ONLY on training data to avoid data leakage.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 8. SAVE PROCESSED TEST CSV
# ---------------------------------------------------------
# IMPORTANT:
# The Streamlit app expects the same processed feature columns that
# the scaler/model were trained on, plus the actual Response column.

# Save the UN-SCALED test features.
# The Streamlit app loads scaler.pkl and performs the scaling before prediction.
test_data = X_test.copy()
test_data["Response"] = y_test.values

# Save the requested test.csv.
test_data.to_csv("test.csv", index=False)

# Also save test_data.csv because the Streamlit app uses that name.
test_data.to_csv("test_data.csv", index=False)

print("\nCreated:")
print("  test.csv")
print("  test_data.csv")


# Save feature names so the application can reproduce the exact order.
os.makedirs("model", exist_ok=True)

joblib.dump(
    list(X.columns),
    "model/feature_columns.pkl"
)

joblib.dump(
    scaler,
    "model/scaler.pkl"
)


# ---------------------------------------------------------
# 9. INITIALIZE FIVE REQUIRED MODELS
# ---------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsClassifier(),

    "Gaussian Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
}


# ---------------------------------------------------------
# 10. TRAIN, EVALUATE AND SAVE MODELS
# ---------------------------------------------------------
results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_scaled, y_train)

    # Use filenames compatible with the Streamlit app.
    filename_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KNN": "knn.pkl",
        "Gaussian Naive Bayes": "naive_bayes.pkl",
        "Random Forest": "random_forest.pkl",
    }

    model_path = os.path.join(
        "model",
        filename_map[name]
    )

    joblib.dump(model, model_path)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Probability for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = roc_auc_score(y_test, y_pred)

    # Required metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy,
        "AUC Score": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC Score": mcc,
    })

    print(
        f"{name}: "
        f"Accuracy={accuracy:.6f}, "
        f"AUC={auc:.6f}, "
        f"Precision={precision:.6f}, "
        f"Recall={recall:.6f}, "
        f"F1={f1:.6f}, "
        f"MCC={mcc:.6f}"
    )


# ---------------------------------------------------------
# 11. PRINT README TABLE
# ---------------------------------------------------------
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("METRICS FOR README.MD")
print("=" * 80)

print(
    results_df.to_markdown(
        index=False,
        floatfmt=".6f"
    )
)


# ---------------------------------------------------------
# 12. SAVE RESULTS
# ---------------------------------------------------------
results_df.to_csv(
    "model_results.csv",
    index=False
)

print("\nGenerated files:")
print("  test.csv")
print("  test_data.csv")
print("  model_results.csv")
print("  model/scaler.pkl")
print("  model/feature_columns.pkl")
print("  model/logistic_regression.pkl")
print("  model/decision_tree.pkl")
print("  model/knn.pkl")
print("  model/naive_bayes.pkl")
print("  model/random_forest.pkl")

print("\nTraining completed successfully.")
