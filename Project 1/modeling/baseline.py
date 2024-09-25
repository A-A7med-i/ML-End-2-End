from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
import numpy as np


class BaseLine:
    """
    A baseline classification model using Logistic Regression.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initializes the class with training data (X) and labels (y).

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self.R_S = 42
        self.X = X
        self.y = y
        self.model = LogisticRegression(
            random_state=self.R_S, max_iter=1000, solver="saga", tol=1e-4
        )



    def train(self, test_size: float = 0.2) -> None:
        """
        Splits data into training and testing sets, trains the model, and stores predictions.

        Args:
            test_size (float, optional): Split ratio for test set. Defaults to 0.2.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.R_S
        )

        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)



    def prediction(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts class labels for a given data point (x).

        Args:
            x (np.ndarray): Data point for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        return self.model.predict(x)




    def metrics(self, y_prediction: np.ndarray) -> dict:
        """
        Calculates and returns various performance metrics for the model.

        Args:
            y_prediction (np.ndarray): Predicted class labels.

        Returns:
            dict: Dictionary containing accuracy, recall, precision, F1-score, and confusion matrix.
        """

        self.acc = accuracy_score(self.y_test, self.y_pred)

        self.recall = recall_score(self.y_test, self.y_pred)

        self.precision = precision_score(self.y_test, self.y_pred)

        self.f1 = f1_score(self.y_test, self.y_pred)

        self.conf_matrix = confusion_matrix(self.y_test, self.y_pred)

        return {
            "accuracy": self.acc,
            "recall": self.recall,
            "precision": self.precision,
            "f1_score": self.f1,
            "confusion_matrix": self.conf_matrix,
        }




    def display_metrics(self) -> None:
        """
        Prints various performance metrics for the model on the test set.
        """

        self.metrics(self.y_pred)

        print(f"Accuracy: {self.acc:.2f}")
        print(f"Recall: {self.recall:.2f}")
        print(f"Precision: {self.precision:.2f}")
        print(f"F1-score: {self.f1:.2f}")
        print("Confusion Matrix:")
        print(self.conf_matrix)




    def run(self) -> None:
        """
        Trains the model, calculates metrics, and displays them.
        """
        self.train()
        self.display_metrics()
