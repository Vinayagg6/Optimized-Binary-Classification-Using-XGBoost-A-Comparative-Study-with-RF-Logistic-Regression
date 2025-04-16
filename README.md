**Project Description**
This project involves predicting a binary outcome C based on 22 input features. Three machine learning models were evaluated:

Logistic Regression,
Random Forest Classifier,
XGBoost Classifier (Final chosen model)

**Final Model: XGBoost**
XGBoost gave the best balance between precision, recall, and F1-score, particularly for the minority class (label 1).

| Model              | Accuracy | Class 1 Recall | Class 1 F1 |
|--------------------|----------|----------------|------------|
| Logistic Regression| 0.57     | 0.89           | 0.51       |
| Random Forest      | 0.75     | 0.03           | 0.05       |
| **XGBoost**        | 0.63     | 0.74           | 0.49       |

**Step-by-Step Process**

**1. Load the Dataset**
Loaded both training and testing data from text files.

Performed basic exploration to check for:

Missing values

Data types

Target class distribution

This step helped understand the structure and quality of the data.

**2. Separate Features and Target**
Extracted the input features (22 columns) and stored the target variable C separately.

This setup is necessary for supervised learning where we predict the target based on features.

**3. Encode the Target Variable**
The target C was categorical (e.g., class labels like A, B, C).

Converted it into numeric labels using label encoding, since most ML models require numeric inputs.

**4. Normalize the Features**
Standardized the feature values so that they’re all on the same scale.

This improves model performance, especially for models like Logistic Regression.

**5. Train Multiple Models**
Trained three different classification models:

Random Forest

Logistic Regression

XGBoost

The goal was to compare their performance and see which one fits best.

**6. Evaluate the Models**
Evaluated all models using the test dataset.

Used metrics such as:

Accuracy: Overall correctness of the model

Precision & Recall: How well the model performs on each class

F1-Score: Balance between precision and recall

This step helped identify the strengths and weaknesses of each model.

**7. Compare and Select the Best Model**
Based on evaluation metrics, XGBoost performed best.

It showed higher accuracy and better balance across all classes.

Chose XGBoost as the final model for the project.

**8. Save the Final Model**
Saved the trained XGBoost model so it can be reused later without retraining.

This is useful for deployment or further testing.


**Libraries Used**
pandas, scikit-learn, xgboost, matplotlib, numpy

 **Learnings**
Handling class imbalance using scale_pos_weight
Comparing models based on real business-driven metrics
Importance of recall over accuracy in imbalanced datasets
