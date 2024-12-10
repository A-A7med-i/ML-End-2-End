import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.preprocessing import LabelEncoder


class Encoder:
    """
    A class for encoding categorical variables.

    Attributes:
        label_encoder: An instance of LabelEncoder for encoding categorical data.
        is_trained: A boolean flag indicating whether the encoder has been trained.
    """

    def __init__(self):
        """
        Initializes the Encoder with a LabelEncoder instance.
        """
        self.label_encoder = LabelEncoder()
        self.is_trained = False

    def train(self, data: pd.DataFrame, target_column: str) -> None:
        """
        Trains the encoder on the given data.

        Args:
            data: The DataFrame containing the data.
            target_column: The name of the column to encode.

        Raises:
            KeyError: If the target_column is not found in the DataFrame.
            ValueError: If the DataFrame is empty.
        """
        if target_column not in data.columns:
            raise KeyError(f"Column '{target_column}' not found in DataFrame")

        if data.empty:
            raise ValueError("Empty dataframe")

        self.label_encoder.fit(data[target_column])
        self.is_trained = True

    def transform(self, input_data: Union[str, List[str], np.ndarray]) -> np.ndarray:
        """
        Transforms the input data using the trained encoder.

        Args:
            input_data: The data to be encoded.

        Returns:
            An array of encoded values.

        Raises:
            ValueError: If the encoder has not been trained.
        """
        if not self.is_trained:
            raise ValueError("Not trained")

        return self.label_encoder.transform(input_data)

    @classmethod
    def is_android(cls, operating_system: str) -> int:
        """
        Checks if the given operating system is Android.

        Args:
            operating_system: The operating system string.

        Returns:
            1 if the operating system is Android, 0 otherwise.
        """
        return 1 if operating_system == "Android" else 0
