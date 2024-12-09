from sklearn.preprocessing import LabelEncoder
from typing import Union
import pandas as pd
import numpy as np


country_dict = {
    "US": "United States",
    "GB": "United Kingdom",
    "CA": "Canada",
    "ES": "Spain",
    "IN": "India",
    "DE": "Germany",
    "FR": "France",
}


class CategoryEncoder:
    """
    A class for encoding categorical features using LabelEncoder.

    Attributes:
        encoder: An instance of LabelEncoder.
    """

    def __init__(self):
        """
        Initializes the CategoryEncoder with a LabelEncoder instance.
        """
        self.encoder = LabelEncoder()

    def fit(self, data: pd.DataFrame, category_column: str) -> None:
        """
        Fits the LabelEncoder on the specified categorical column.

        Args:
            data: The DataFrame containing the data.
            category_column: The name of the categorical column.
        """
        self.encoder.fit(data[category_column])

    def encode(self, values: Union[pd.Series, list, np.ndarray]) -> np.ndarray:
        """
        Encodes the given values using the fitted LabelEncoder.

        Args:
            values: The values to be encoded.

        Returns:
            An array of encoded values.
        """
        return self.encoder.transform(values)


def replace_country_codes(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Replaces country codes with their full names in the given DataFrame.

    Args:
        data: The DataFrame containing the country codes.
        column: The name of the column containing the country codes.

    Returns:
        A new DataFrame with country codes replaced by full names.
    """
    data[column] = data[column].map(country_dict).fillna("Other")

    return data
