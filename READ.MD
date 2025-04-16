# Binary Classification with XGBoost | Predicting Target C

## 📁 Project Description
This project involves predicting a binary outcome `C` based on 22 input features. Three machine learning models were evaluated:
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier (Final chosen model)

##  Final Model: XGBoost
XGBoost gave the best balance between precision, recall, and F1-score, particularly for the minority class (label 1).

| Model              | Accuracy | Class 1 Recall | Class 1 F1 |
|--------------------|----------|----------------|------------|
| Logistic Regression| 0.57     | 0.89           | 0.51       |
| Random Forest      | 0.75     | 0.03           | 0.05       |
| **XGBoost**        | 0.63     | 0.74           | 0.49       |

##  Key Steps
- Data cleaning and preprocessing (handling dates, scaling)
- Model training and comparison
- Feature importance analysis
- Saving and exporting predictions

##  Files
- `model_comparison.ipynb` - Notebooks comparing all 3 models
- `final_model_xgb.py` - Final version using only XGBoost
- `results/` - Output files, plots, and predictions

##  Libraries Used
- `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `numpy`

---

## 🧠 Learnings
- Handling class imbalance using `scale_pos_weight`
- Comparing models based on real business-driven metrics
- Importance of recall over accuracy in imbalanced datasets
