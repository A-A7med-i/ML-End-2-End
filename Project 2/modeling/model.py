import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RANDOM_STATE = 42


@dataclass
class ModelMetrics:
    """
    A dataclass to store model evaluation metrics.

    Attributes:
        mae: Mean Absolute Error
        mse: Mean Squared Error
        rmse: Root Mean Squared Error
        r2: R-squared score
    """

    mae: float
    mse: float
    rmse: float
    r2: float


class Agent:
    """
    A class to encapsulate the training, prediction, and evaluation of a machine learning model.

    Attributes:
        model: The underlying machine learning model (e.g., a scikit-learn estimator).
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target vector.
        y_test: Test target vector.
        y_pred: Predicted target values.
        metrics: Model evaluation metrics.
    """

    def __init__(self, model: BaseEstimator):
        """
        Initializes the Agent with the given model.

        Args:
            model: The machine learning model to use.
        """

        self.model = model
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.metrics = None

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> None:
        """
        Splits the data into training and testing sets and trains the model.

        Args:
            X: Feature matrix.
            y: Target vector.
            test_size: Proportion of data to use for testing. Defaults to 0.2.
        """

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE
        )
        self.model.fit(self.X_train, self.y_train)

    def prediction(self) -> np.ndarray:
        """
        Makes predictions on the test data using the trained model.

        Returns:
            np.ndarray: Predicted target values.
        """

        self.y_pred = self.model.predict(self.X_test)

        return self.y_pred

    def evaluate(self, y_pred: np.ndarray) -> ModelMetrics:
        """
        Calculates and returns the model evaluation metrics.

        Returns:
            ModelMetrics: A dataclass containing the MAE, MSE, RMSE, and R-squared scores.
        """

        self.metrics = ModelMetrics(
            mae=mean_absolute_error(self.y_test, y_pred),
            mse=mean_squared_error(self.y_test, y_pred),
            rmse=np.sqrt(mean_squared_error(self.y_test, y_pred)),
            r2=r2_score(self.y_test, y_pred),
        )
        return self.metrics

    @staticmethod
    def display_metrics(metrics: ModelMetrics) -> None:
        """
        Displays all model evaluation metrics in a formatted way.

        Args:
            metrics: A ModelMetrics object containing the evaluation metrics.
        """

        print(f"Model Performance Metrics:")
        print("-" * 25)
        print(f"Mean Absolute Error (MAE): {metrics.mae:.2f}")
        print(f"Mean Squared Error (MSE): {metrics.mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {metrics.rmse:.2f}")
        print(f"R-squared (R2): {metrics.r2:.2f}")
