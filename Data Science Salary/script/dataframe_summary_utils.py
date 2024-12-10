import pandas as pd


class DataFrameSummary:
    """
    A class to generate summary statistics for a pandas DataFrame.

    Attributes:
        df: The input pandas DataFrame.
        df_summary: A pandas DataFrame to store the summary statistics.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Initializes the DataFrameSummary object.

        Args:
            df: The input pandas DataFrame.
        """
        self.df = df
        self.df_summary = pd.DataFrame()

    def get_column_types(self) -> pd.Series:
        """
        Gets the data types of each column in the DataFrame.

        Returns:
            A pandas Series containing the data types of each column.
        """
        d_types = self.df.dtypes
        return d_types

    def calculate_max_values(self) -> list:
        """
        Calculates the maximum value for each column.

        Returns:
            A list containing the maximum value for each column.
            For object columns, "No Max" is returned.
        """
        max = [
            (
                self.df[column].max().round(3)
                if self.df[column].dtype != "object"
                else "No Max"
            )
            for column in self.df.columns
        ]

        return max

    def calculate_min_values(self) -> list:
        """
        Calculates the minimum value for each column.

        Returns:
            A list containing the minimum value for each column.
            For object columns, "No Min" is returned.
        """
        min = [
            (
                self.df[column].min().round(3)
                if self.df[column].dtype != "object"
                else "No Min"
            )
            for column in self.df.columns
        ]

        return min

    def calculate_std_values(self) -> list:
        """
        Calculates the standard deviation for each column.

        Returns:
            A list containing the standard deviation for each column.
            For object columns, "No Std" is returned.
        """
        std = [
            (
                self.df[column].std().round(3)
                if self.df[column].dtype != "object"
                else "No Std"
            )
            for column in self.df.columns
        ]

        return std

    def calculate_mean_values(self) -> list:
        """
        Calculates the mean value for each column.

        Returns:
            A list containing the mean value for each column.
            For object columns, "No Mean" is returned.
        """
        mean = [
            (
                self.df[column].mean().round(3)
                if self.df[column].dtype != "object"
                else "No Mean"
            )
            for column in self.df.columns
        ]
        return mean

    def calculate_median_values(self) -> list:
        """
        Calculates the median value for each column.

        Returns:
            A list containing the median value for each column.
            For object columns, "No Median" is returned.
        """
        median = [
            (
                self.df[column].median().round(3)
                if self.df[column].dtype != "object"
                else "No Median"
            )
            for column in self.df.columns
        ]
        return median

    def count_unique_values(self) -> list:
        """
        Counts the number of unique values for each column.

        Returns:
            A list containing the number of unique values for each column.
        """
        unique = [
            (
                len(self.df[column].unique())
                if self.df[column].dtype == "object"
                else self.df[column].nunique()
            )
            for column in self.df.columns
        ]
        return unique

    def count_nulls(self) -> list:
        """
        Counts the number of null values for each column.

        Returns:
            A list containing the number of null values for each column.
        """
        nulls = self.df.isnull().sum().to_list()

        return nulls

    def generate_dataframe_summary(self) -> pd.DataFrame:
        """
        Generates a pandas DataFrame containing the summary statistics.

        Returns:
            A pandas DataFrame with columns for data type, min, max, std, mean, median,
            unique values, and null counts for each column.
        """
        self.df_summary["Data Type"] = self.get_column_types()

        self.df_summary["Min"] = self.calculate_min_values()

        self.df_summary["Max"] = self.calculate_max_values()

        self.df_summary["Std"] = self.calculate_std_values()

        self.df_summary["Mean"] = self.calculate_mean_values()

        self.df_summary["Median"] = self.calculate_median_values()

        self.df_summary["Unique Values"] = self.count_unique_values()

        self.df_summary["Nulls"] = self.count_nulls()

        return self.df_summary
