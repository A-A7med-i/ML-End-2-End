from .models import ClassificationModel
from sklearn.svm import SVC
from typing import Union
import numpy as np
import pandas as pd

MODEL = SVC()


class SupportVectorClassifier(ClassificationModel):
    """
    A class that extends the ClassificationModel for Support Vector Classification.

    Inherits from the ClassificationModel and provides a specific method
    for client-side prediction.
    """

    def __init__(self):
        """
        Initializes the SupportVectorClassifier.
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
