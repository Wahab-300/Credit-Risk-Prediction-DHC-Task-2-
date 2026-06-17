# DevelopersHub Data Science & Analytics Internship

## Task 2: Credit Risk Prediction

### Objective
Predict whether a loan applicant is likely to default on a loan using Machine Learning.

### Dataset
- **Source:** Kaggle Loan Prediction Dataset
- **Rows:** 614 | **Columns:** 13
- **Target:** Loan_Status (Y = Approved, N = Rejected)

### Approach
- Handled missing values using median (numerical) and mode (categorical)
- Visualized Loan Amount, Education, Income, and Loan Status distributions
- Encoded categorical features using Label Encoding
- Scaled features using StandardScaler
- Trained Logistic Regression model (80/20 split)

### Results
| Metric | Value |
|--------|-------|
| Accuracy | 78.86% |
| Correctly Approved | 79 |
| Correctly Rejected | 18 |
| Misclassified | 26 |

### Key Insights
- Credit history is the strongest predictor of loan approval
- Model struggles with rejected loans due to class imbalance
- Graduates have higher loan approval rate than non-graduates

### Libraries Used
`pandas` `matplotlib` `seaborn` `scikit-learn`

---
*DevelopersHub Corporation — Data Science & Analytics Internship*
