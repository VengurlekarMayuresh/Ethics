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

    # Define which columns are numeric (for validation and log transforms)
    numeric_features = [
        'ApplicantIncome',
        'CoapplicantIncome',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History'
    ]
    categorical_features = [
        'Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'Property_Area'
    ]

    def __init__(self):
        self.raw_feature_names = None
        # Define derived features for identification
        self.derived_features = [
            'Total_Income',
            'ApplicantIncomeLog',
            'CoapplicantIncomeLog',
            'LoanAmountLog',
            'Loan_Amount_Term_Log',
            'Total_IncomeLog'
        ]

    def fit(self, X, y=None):
        """Store raw feature names for schema extraction."""
        if hasattr(X, 'columns'):
            self.raw_feature_names = list(X.columns)
        else:
            # Fallback to class defaults if fit with numpy array
            self.raw_feature_names = self.numeric_features + self.categorical_features
        return self

    def get_feature_names_out(self, input_features=None):
        """Standard sklearn method for feature names after transformation."""
        raw = input_features if input_features is not None else self.raw_feature_names
        if raw is None:
            raw = self.numeric_features + self.categorical_features
        return list(raw) + self.derived_features

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

        # Ensure all numeric features exist and are of numeric type
        # This is critical for SHAP which may pass mixed-type numpy arrays
        for col in self.numeric_features:
            if col not in X_df.columns:
                raise ValueError(f"Missing required feature: {col}")
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

        # Create derived features
        X_df['Total_Income'] = X_df['ApplicantIncome'] + X_df['CoapplicantIncome']
        
        # Ensure Total_Income is also numeric before log transforms
        X_df['Total_Income'] = pd.to_numeric(X_df['Total_Income'], errors='coerce')
        
        X_df['ApplicantIncomeLog'] = np.log1p(X_df['ApplicantIncome'].astype(float))
        X_df['CoapplicantIncomeLog'] = np.log1p(X_df['CoapplicantIncome'].astype(float))
        X_df['LoanAmountLog'] = np.log1p(X_df['LoanAmount'].astype(float))
        X_df['Loan_Amount_Term_Log'] = np.log1p(X_df['Loan_Amount_Term'].astype(float))
        X_df['Total_IncomeLog'] = np.log1p(X_df['Total_Income'].astype(float))

        return X_df
