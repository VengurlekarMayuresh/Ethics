"""
Loan Prediction Custom Transformers
These classes are used in the loan_prediction_pipeline.pkl and must be importable
by the backend during model loading (via PICKLE_CLASS_MODULES).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that creates derived features from raw numeric columns.
    This transformer is part of the loan prediction pipeline.

    Expected RAW input columns (user provides these):
        Categorical: Gender, Married, Dependents, Education, Self_Employed, Property_Area
        Numeric: ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

    Output columns (after transform):
        All original raw columns + these derived numeric features:
        - Total_Income
        - ApplicantIncomeLog
        - CoapplicantIncomeLog
        - LoanAmountLog
        - Loan_Amount_Term_Log
        - Total_IncomeLog
    """

    def __init__(self):
        self.raw_feature_names = None
        # Define which columns are numeric (for validation and log transforms)
        self.numeric_features = [
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History'
        ]
        self.categorical_features = [
            'Gender',
            'Married',
            'Dependents',
            'Education',
            'Self_Employed',
            'Property_Area'
        ]

    def fit(self, X, y=None):
        """Store raw feature names for schema extraction."""
        if hasattr(X, 'columns'):
            self.raw_feature_names = list(X.columns)
        return self

    def transform(self, X):
        """
        Transform raw input X by adding derived features.
        Input: DataFrame or array with raw numeric_features + categorical_features
        Output: DataFrame with all features (original + derived)
        """
        if isinstance(X, np.ndarray):
            # Convert to DataFrame with proper column names
            X_df = pd.DataFrame(
                X,
                columns=self.numeric_features + self.categorical_features
            )
        else:
            X_df = X.copy()

        # Ensure all numeric features exist
        for col in self.numeric_features:
            if col not in X_df.columns:
                raise ValueError(f"Missing required feature: {col}")

        # Create derived features
        X_df['Total_Income'] = X_df['ApplicantIncome'] + X_df['CoapplicantIncome']
        X_df['ApplicantIncomeLog'] = np.log(X_df['ApplicantIncome'] + 1)
        X_df['CoapplicantIncomeLog'] = np.log(X_df['CoapplicantIncome'] + 1)
        X_df['LoanAmountLog'] = np.log(X_df['LoanAmount'] + 1)
        X_df['Loan_Amount_Term_Log'] = np.log(X_df['Loan_Amount_Term'] + 1)
        X_df['Total_IncomeLog'] = np.log(X_df['Total_Income'] + 1)

        return X_df
