from sklearn.preprocessing import LabelEncoder
from typing import List, Dict
import pandas as pd


class CategoryEncoder:
    """
    A class for encoding categorical features in a DataFrame.

    Attributes:
        encoders: A dictionary to store LabelEncoder instances for each categorical column.
        encoded_columns: A list of the encoded columns.
    """

    def __init__(self):
        """
        Initializes the CategoryEncoder with empty dictionaries and lists.
        """
        self.encoders: dict = {}
        self.encoded_columns: List[str] = []

    def fit(
        self, data: pd.DataFrame, categorical_columns: List[str]
    ) -> "CategoryEncoder":
        """
        Fits the LabelEncoder for each specified categorical column.

        Args:
            data: The DataFrame containing the data.
            categorical_columns: A list of the names of the categorical columns.

        Returns:
            The fitted CategoryEncoder object.
        """
        for column in categorical_columns:
            self.encoders[column] = LabelEncoder().fit(data[column])

        self.encoded_columns = categorical_columns

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the given DataFrame by encoding the specified categorical columns.

        Args:
            data: The DataFrame to be transformed.

        Returns:
            A new DataFrame with the specified categorical columns encoded.
        """
        data_encoded = data.copy()
        for column in self.encoded_columns:
            data_encoded[column] = self.encoders[column].transform(data[column])

        return data_encoded

    def fit_transform(
        self, data: pd.DataFrame, categorical_columns: List[str]
    ) -> pd.DataFrame:
        """
        Fits the encoder and then transforms the given DataFrame.

        Args:
            data: The DataFrame to be transformed.
            categorical_columns: A list of the names of the categorical columns.

        Returns:
            A new DataFrame with the specified categorical columns encoded.
        """
        return self.fit(data, categorical_columns).transform(data)


def target_encoding(value: str) -> int:
    """
    Encodes target labels to numerical values.

    Args:
        value: The target label ("Normal", "Inconclusive", or "Abnormal").

    Returns:
        0 for "Normal", 1 for "Inconclusive", and 2 for "Abnormal".
    """
    if value == "Normal":
        return 0
    elif value == "Inconclusive":
        return 1
    elif value == "Abnormal":
        return 2
