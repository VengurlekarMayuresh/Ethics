import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create engineered features:
    - Log transforms of ApplicantIncome, LoanAmount, Loan_Amount_Term
    - Total_Income and its log
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Log transforms
        X_copy["ApplicantIncomeLog"] = np.log(X_copy["ApplicantIncome"] + 1)
        X_copy["LoanAmountLog"] = np.log(X_copy["LoanAmount"] + 1)
        X_copy["Loan_Amount_Term_Log"] = np.log(X_copy["Loan_Amount_Term"] + 1)

        # Total income
        X_copy["Total_Income"] = X_copy["ApplicantIncome"] + X_copy["CoapplicantIncome"]
        X_copy["Total_Income_Log"] = np.log(X_copy["Total_Income"] + 1)

        return X_copy