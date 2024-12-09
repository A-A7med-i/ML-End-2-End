import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from scripts.configuration import RANDOM_STATE
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
from typing import Union
import pandas as pd


@dataclass
class EvaluationMetrics:
    """
    Holds evaluation metrics for a classification model.

    Attributes:
        accuracy: Accuracy score.
        recall: Recall score.
        precision: Precision score.
        f1: F1-score.
        confusion_matrix: Confusion matrix.
    """

    accuracy: float
    recall: float
    precision: float
    f1: float
    confusion_matrix: np.ndarray

    def display(self) -> None:
        """Displays classification model evaluation metrics in a formatted way."""

        print(f"Model Performance Metrics:")
        print("-" * 25)
        print(f"Accuracy: {self.accuracy:.2f}")
        print(f"Recall: {self.recall:.2f}")
        print(f"Precision: {self.precision:.2f}")
        print(f"F1-Score: {self.f1:.2f}")
        print("\nConfusion Matrix:")
        print(self.confusion_matrix)


class BaseModel:
    """
    A wrapper class for classification models.

    Attributes:
        classifier: The underlying classification model (e.g., from scikit-learn).
        X_train: Training features.
        X_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
        y_pred: Predicted labels.
        evaluation_metrics: Evaluation metrics for the model.
    """

    def __init__(self, classifier: BaseEstimator):
        """
        Initializes the ClassificationModel.

        Args:
            classifier: A scikit-learn classification model.
        """
        self.classifier = classifier
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.evaluation_metrics = None

    def train(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]
    ) -> None:
        """
        Trains the classification model on the provided data.

        Args:
            X: The training features.
            y: The training labels.
        """

        self._split_data(X, y)
        self.classifier.fit(self.X_train, self.y_train)

    def _split_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = 0.2,
    ) -> None:
        """Splits data into training and testing sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )

    def prediction(self) -> np.ndarray:
        """
        Predicts labels for the testing data.

        Returns:
            The predicted labels.
        """

        self.y_pred = self.classifier.predict(self.X_test)
        return self.y_pred

    def evaluate(self) -> EvaluationMetrics:
        """
        Evaluates the model performance on the testing data.

        Returns:
            An EvaluationMetrics object containing the evaluation metrics.
        """

        self.evaluation_metrics = EvaluationMetrics(
            accuracy=round(accuracy_score(self.y_test, self.y_pred), 3),
            recall=recall_score(self.y_test, self.y_pred, average="micro").round(3),
            precision=precision_score(self.y_test, self.y_pred, average="micro").round(
                3
            ),
            f1=f1_score(self.y_test, self.y_pred, average="micro").round(3),
            confusion_matrix=confusion_matrix(self.y_test, self.y_pred),
        )

        return self.evaluation_metrics
