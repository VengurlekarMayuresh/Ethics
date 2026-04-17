import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create engineered features:
    - Log transforms of ApplicantIncome, LoanAmount, Loan_Amount_Term
    - Total_Income and its log
    """

    # Define features for robust schema extraction
    numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    derived_features = [
        'ApplicantIncomeLog', 
        'CoapplicantIncomeLog', 
        'LoanAmountLog', 
        'Loan_Amount_Term_Log', 
        'Total_Income', 
        'Total_IncomeLog',
        'Total_Income_Log'
    ]

    def __init__(self):
        self.raw_feature_names = None

    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.raw_feature_names = list(X.columns)
        else:
            self.raw_feature_names = self.numeric_features + self.categorical_features
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            # Convert to DataFrame with proper column names
            X_df = pd.DataFrame(
                X,
                columns=self.numeric_features + self.categorical_features
            )
        else:
            X_df = X.copy()

        # Ensure all numeric features are of numeric type
        for col in self.numeric_features:
            if col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

        # Log transforms
        X_df["ApplicantIncomeLog"] = np.log1p(X_df["ApplicantIncome"].astype(float))
        X_df["CoapplicantIncomeLog"] = np.log1p(X_df["CoapplicantIncome"].astype(float))
        X_df["LoanAmountLog"] = np.log1p(X_df["LoanAmount"].astype(float))
        X_df["Loan_Amount_Term_Log"] = np.log1p(X_df["Loan_Amount_Term"].astype(float))

        # Total income
        X_df["Total_Income"] = X_df["ApplicantIncome"] + X_df["CoapplicantIncome"]
        X_df["Total_Income"] = pd.to_numeric(X_df["Total_Income"], errors='coerce')
        X_df["Total_IncomeLog"] = np.log1p(X_df["Total_Income"].astype(float))
        X_df["Total_Income_Log"] = X_df["Total_IncomeLog"]

        return X_df