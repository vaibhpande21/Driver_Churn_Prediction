# Churn Prediction Model

## Project Overview
This project focuses on predicting customer churn using various machine learning algorithms. The dataset used contains features related to drivers' behaviors, and the goal is to predict whether a driver will churn (leave the service) or not. Multiple models, including Decision Tree, Random Forest, Gradient Boosting, XGBoost, and LightGBM, are evaluated to determine the best performing model.

## Objective
- Predict driver churn with high recall (minimizing Type 2 errors).
- Evaluate different machine learning models to find the most effective one in terms of accuracy, recall, precision, and F1 score.
- Handle class imbalance using SMOTE and perform data scaling with Standard Scaler.

## Data Preprocessing
- **Class Imbalance Handling:** SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the dataset by creating synthetic samples for the minority class (churned drivers).
- **Data Scaling:** StandardScaler is applied to scale the features before feeding them into machine learning models.

## Models Used
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Gradient Boosting Decision Tree (GBDT)**
- **XGBoost Classifier**
- **LightGBM Classifier**

### Model Evaluation
The models are evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC Score
- ROC Curve

## Steps
1. **Data Preprocessing**
    - Load dataset and handle missing values.
    - Use SMOTE to address class imbalance.
    - Apply StandardScaler for feature scaling.
  
2. **Model Building**
    - Train and evaluate models: Decision Tree, Random Forest, GBDT, XGBoost, and LightGBM.
    - Perform hyperparameter tuning using GridSearchCV.
    - Compare models based on AUC, precision, recall, and F1 score.
  
3. **Model Evaluation**
    - Visualize confusion matrices for each model.
    - Plot ROC and Precision-Recall curves.
    - Display classification reports.

4. **Results**
    - The best model (LightGBM) achieved an accuracy of 82% and an F1 score of 83%.

## Business Insights and Recommendations

### 1. **Target High-Risk Drivers:**
   - **Insight:** Males and drivers in Grades 1 and 2 are more likely to churn.
   - **Recommendation:** Implement retention strategies for male drivers and those with lower grades, such as offering performance incentives or career growth opportunities.

### 2. **Focus on Income Levels:**
   - **Insight:** Drivers with income between ₹40,000 and ₹70,000 have higher churn rates.
   - **Recommendation:** Introduce income-based incentives or bonuses for drivers in this income bracket to encourage retention.

### 3. **Boost Performance Ratings:**
   - **Insight:** Quarterly rating improvements correlate with lower churn.
   - **Recommendation:** Invest in training programs and performance feedback to help drivers improve their ratings, leading to higher job satisfaction and retention.

### 4. **Monitor Business Value:**
   - **Insight:** Drivers with lower business value are more likely to churn.
   - **Recommendation:** Develop strategies to increase business for low-performing drivers, such as pairing them with higher-demand routes or providing targeted marketing efforts.

### 5. **Enhance Driver Experience Based on City:**
   - **Insight:** Churn rates vary by city, indicating location-specific issues.
   - **Recommendation:** Tailor retention strategies based on city-specific needs, such as adjusting pay rates or providing better support in cities with higher churn.

## Dependencies

- Python 3.x
- Pandas
- Numpy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn

## Acknowledgments
- Libraries: Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn

