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

https://github.com/2025ac05880-harikiran/AIMLCZC565_MLassignment2/tree/main

## << Streamlit app Link >>

<< TO BE UPDATED>>


## d. Models Used & Evaluation Metrics

The data partition was passed through five distinct classifiers using k-fold cross-validation. The cross-validation performance averages for each metric are outlined below:

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.877679 | 0.864910 | 0.862326 | 0.877679 | 0.863364 | 0.435943 |
| **Decision Tree** | 0.860714 | 0.710113 | 0.837675 | 0.860714 | 0.841585 | 0.337396 |
| **K-Nearest Neighbor (KNN)** | 0.856696 | 0.723195 | 0.834008 | 0.856696 | 0.839616 | 0.329150 |
| **Gaussian Naive Bayes** | 0.681696 | 0.779384 | 0.841264 | 0.681696 | 0.713623 | 0.300602 |
| **Random Forest** | 0.875446 | 0.870216 | 0.861503 | 0.875446 | 0.848002 | 0.383906 |

## e. Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Showed the highest Accuracy (~0.878), Recall (~0.878), and MCC (~0.436) among the five models. Its F1 Score (~0.863) was also strong, indicating a good balance between Precision and Recall. The model provides a strong, interpretable baseline for identifying customers likely to respond to the marketing campaign. |
| **Decision Tree** | Achieved strong Accuracy (~0.861) and Recall (~0.861), with an F1 Score of ~0.842. However, its AUC (~0.710) and MCC (~0.337) were substantially lower than those of Logistic Regression and Random Forest, suggesting weaker overall class discrimination and generalization. |
| **KNN** | Delivered a stable performance with Accuracy (~0.857), Precision (~0.834), Recall (~0.857), and F1 Score (~0.840). Its AUC (~0.723) was higher than the Decision Tree but still considerably below Logistic Regression and Random Forest, indicating moderate ability to distinguish between responding and non-responding customers. |
| **Gaussian Naive Bayes** | Produced the weakest overall results, with Accuracy and Recall of ~0.682 and an F1 Score of ~0.714. Although its Precision (~0.841) remained relatively high, its low Recall indicates that the model struggled to identify a substantial portion of the positive-response cases. |
| **Random Forest (Ensemble)** | Delivered performance very close to Logistic Regression, with Accuracy (~0.875), Precision (~0.862), Recall (~0.875), and the highest F1 Score (~0.848) among all models. It also achieved the highest AUC (~0.870), demonstrating the strongest overall ranking and class-discrimination capability. |
| **Overall Winner** | **Logistic Regression** is selected as the overall model based on the highest Accuracy (~87.8%) and MCC (~0.436), while also maintaining a strong F1 Score (~0.863). Random Forest is a very close alternative and achieves the best AUC (~0.870) and F1 Score (~0.848), making it particularly attractive when ranking/discrimination capability is prioritized. For this project, Logistic Regression is preferred for its combination of predictive performance, interpretability, and computational efficiency. |

## Overall Project Winner

The **Logistic Regression** model is selected as the overall winner for this marketing campaign classification task. It achieved the highest Accuracy (~87.8%) and MCC (~0.436), while maintaining a strong F1 Score (~0.863). These results indicate that Logistic Regression provides a strong balance of predictive performance, reliability, interpretability, and computational efficiency.

The **Random Forest** model is a close second and deserves consideration for scenarios where ranking and class-discrimination capability are more important. It achieved the highest AUC (~0.870) and the highest F1 Score (~0.848). Therefore, both models are strong candidates, but **Logistic Regression is retained as the final model for this project based on the overall evaluation criteria**.
