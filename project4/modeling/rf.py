from sklearn.ensemble import RandomForestClassifier
from .base import BaseModel
from typing import Union
import pandas as pd
import numpy as np


MODEL = RandomForestClassifier(n_estimators=500, max_depth=20)


class RandomForest(BaseModel):
    """
    A class that extends the BaseModel for Random Forest Classification.

    Inherits from the BaseModel and provides a specific method
    for client-side prediction.
    """

    def __init__(self):
        """
        Initializes the RandomForestClassifier.
        """
        super().__init__(MODEL)

    def predict_client(
        self, client_input: Union[np.ndarray, pd.DataFrame]
    ) -> np.ndarray:
        """
        Predicts the class labels for client input data.

        Args:
            client_input: The input data for prediction.

        Returns:
            An array of predicted class labels.
        """
        return self.classifier.predict(client_input)
