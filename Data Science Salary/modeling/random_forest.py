from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any
from .model import Agent
import numpy as np
import joblib

MODEL = RandomForestRegressor()


class RandomForest(Agent):
    """
    A class that extends the Agent class to handle Random Forest regression.

    Inherits from the Agent class and provides methods for hyperparameter tuning,
    prediction, and model saving specific to Random Forest models.
    """

    def __init__(self):
        """
        Initializes the RandomForest object.
        """
        super().__init__(MODEL)

    def tune_hyperparameters(self, param_grid: Dict[str, Any]) -> RandomForestRegressor:
        """
        Performs hyperparameter tuning using GridSearchCV.

        Args:
            param_grid: A dictionary specifying the parameter grid for GridSearchCV.

        Returns:
            The best estimator found by GridSearchCV.
        """
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring="accuracy", n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)

        best_model = grid_search.best_estimator_

        return best_model

    def prediction(self, model: RandomForestRegressor) -> np.ndarray:
        """
        Makes predictions using the given model.

        Args:
            model: The trained model to use for prediction.

        Returns:
            The predicted values.
        """

        return model.predict(self.X_test)

    @staticmethod
    def save_model(model: RandomForestRegressor, path: str) -> None:
        """
        Saves the given model to the specified path using joblib.

        Args:
            model: The model to be saved.
            path: The path where the model should be saved.
        """
        joblib.dump(model, path)
