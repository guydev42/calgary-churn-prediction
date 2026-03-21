# Telecom customer churn prediction

## Problem statement

Customer churn costs telecom companies billions in lost revenue each year. This project builds a predictive model to identify customers at high risk of cancelling their service, enabling targeted retention campaigns that maximize ROI.

Using a dataset of 5,000 telecom customers with 20 features covering demographics, service subscriptions, billing, and account tenure, the project compares four classification models and quantifies the business value of churn intervention.

## Key findings

- **Month-to-month contracts** are the strongest churn predictor, with churn rates 3x higher than two-year contracts
- **Fiber optic customers** churn more than DSL customers, likely due to higher monthly charges and service expectations
- **Electronic check** payment method is associated with significantly higher churn compared to automatic payment methods
- **Short tenure** (under 12 months) combined with no support services creates the highest-risk customer segment
- The optimal intervention threshold produces substantial monthly net savings by balancing intervention costs against retained revenue

## Technical approach

1. **Data preparation**: Missing value imputation, feature engineering (tenure groups, support service count, charges per tenure month), encoding (label + one-hot)
2. **Model comparison**: Logistic Regression, Random Forest, XGBoost, LightGBM with GridSearchCV hyperparameter tuning
3. **Explainability**: SHAP values for global and local feature importance
4. **Business impact**: Cost-benefit analysis with threshold optimization ($50 intervention cost vs. $75/month retained revenue)

## Project structure

```
project_14_customer_churn_prediction/
  data/               Telco churn dataset (5,000 customers)
  src/                Data loading, feature engineering, model training
  models/             Saved best model and scaler
  outputs/            Plots, SHAP values, comparison tables
  notebooks/          Exploratory data analysis notebook
  app.py              Streamlit dashboard (5 pages)
```

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset
python generate_data.py

# Train models and generate outputs
python -c "
from src.data_loader import load_and_prepare
from src.model import train_and_evaluate
X_train, X_test, y_train, y_test, fn = load_and_prepare('data/telco_churn.csv')
train_and_evaluate(X_train, X_test, y_train, y_test, fn)
"

# Launch dashboard
streamlit run app.py
```

## Tech stack

Python, pandas, scikit-learn, XGBoost, LightGBM, SHAP, Streamlit, Plotly, Matplotlib, Seaborn
