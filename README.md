# AIMLCZC565_MLassignment2
Machine Learning Assignment - 2 using model comparison and displays the results in Streamlit application
# Kaggle Customer Personality Analysis Classification

## a. Problem Statement

The objective of this project is to implement multiple machine learning classification models to predict whether a customer will accept a company's marketing campaign offer. The target variable `Response` is a binary classification where a customer either accepts the offer (1) or rejects it (0). This predictive capability enables businesses to optimize marketing expenses by targeting consumers who exhibit the highest statistical probability of responding to specific promotional incentives.

## b. Dataset Description

The predictive models are trained and evaluated on the **Kaggle Customer Personality Analysis Dataset**. The source data captures an expansive footprint of customer behavior across three macro categories:

- **Demographics:** Age (derived from year of birth), education level, marital status, annual household income, and number of children or teenagers living at home.
- **Spending Patterns:** Continuous numerical tracking of monetary amounts spent over a two-year window on six primary product segments: wines, fruits, meat products, fish products, sweet products, and gold products.
- **Engagement Channels:** The counts of purchases completed across multiple avenues including the company website, direct catalog marketing, physical brick-and-mortar storefronts, and discount deal opportunities, alongside the volume of monthly web visits.

The final dataset contains complete transaction histories and customer traits, where rows with critical missing features (such as blank income values) are systematically handled during pre-processing.

## c. Github Repository Link

https://github.com/2025ac05880-harikiran/AIMLCZC565_MLassignment2

## d. Models Used & Evaluation Metrics

The data partition was passed through five distinct classifiers using k-fold cross-validation. The cross-validation performance averages for each metric are outlined below:

| ML Model Name                | Accuracy | AUC Score | Precision | Recall   | F1 Score | MCC Score |
| :--------------------------- | :------- | :-------- | :-------- | :------- | :------- | :-------- |
| **Logistic Regression**      | 0.877679 | 0.864910  | 0.862326  | 0.877679 | 0.863364 | 0.435943  |
| **Decision Tree**            | 0.860714 | 0.710113  | 0.837675  | 0.860714 | 0.841585 | 0.337396  |
| **K-Nearest Neighbor (KNN)** | 0.856696 | 0.723195  | 0.834008 | 0.856696 | 0.839616 | 0.329150  |
| **Gaussian Naive Bayes**     | 0.681696 | 0.779384  | 0.841264 | 0.681696 | 0.713623 | 0.300602  |
| **Random Forest**            | 0.875446 | 0.870216  | 0.861503 | 0.875446 | 0.848002 | 0.383906  |

## e. Observations & Performance Analysis

### 1. Logistic Regression

- **Performance Summary:** Achieved the highest performance across almost all standalone target matrices, including the top baseline Accuracy (~87.8%), Recall (~87.8%), and F1 Score (~0.863).
- **Analytical Insight:** Its Matthews Correlation Coefficient (MCC) lead of 0.436 demonstrates that the decision boundary separates the class imbalances more accurately than competing configurations. This signifies that the normalized demographic markers and linear trends in purchase values scale cleanly without requiring complex tree partitioning.

### 2. Decision Tree

- **Performance Summary:** Delivered an Accuracy of ~86.1% and a matching Recall of ~86.1%.
- **Analytical Insight:** Despite strong accuracy metrics, its Area Under the ROC Curve (AUC Score) lagged noticeably at ~0.710. This indicates lower discrimination threshold versatility, exposing a vulnerability to over-fitting on specific demographic subsets.

### 3. K-Nearest Neighbor (KNN)

- **Performance Summary:** Provided stable baseline indicators with an Accuracy of ~85.7% and an F1 Score of ~0.840.
- **Analytical Insight:** KNN handled the clustering tightly but suffered minor boundary dilution. The geometric distances between target points can become slightly distorted due to high feature counts across multiple non-correlated spending metrics (e.g., matching low gold spend with high wine spend).

### 4. Gaussian Naive Bayes

- **Performance Summary:** Yielded the weakest absolute showing in the matrix with a drop in Accuracy to ~68.2% and a constrained Recall of ~68.2%.
- **Analytical Insight:** Naive Bayes assumes strict independence between variables. This assumption fails significantly in customer personality tracking, as continuous parameters like household income, product spending volumes, and purchase channels are naturally heavily dependent on one another.

### 5. Random Forest (Ensemble)

- **Performance Summary:** Competed directly with the top performer, delivering a strong baseline Accuracy of ~87.5% and a comparable Precision of ~0.862.
- **Analytical Insight:** Secured the absolute highest class-discrimination rating with an AUC Score of ~0.870. The ensemble structure successfully smoothed out the tree variances that hindered the singular Decision Tree model.

## Overall Project Winner

The **Logistic Regression** model is chosen as the optimal choice for deployment in this campaign targeting pipeline. It consistently recorded the highest statistical accuracy (~87.8%), optimal balanced error correction (F1: ~0.863), and the most predictive confidence value (MCC: ~0.436). It offers the company a robust, computationally efficient, and highly interpretable engine to maximize marketing conversions.
