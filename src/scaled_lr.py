"""
Scaled Logistic Regression wrapper module

This module defines the ScaledLogisticRegression class that's pickled with our models.
Moving it to its own module ensures proper loading.
"""

class ScaledLogisticRegression:
    """Wrapper that scales inputs before calling a sklearn LogisticRegression."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        return self.model.predict(self.scaler.transform(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(X))