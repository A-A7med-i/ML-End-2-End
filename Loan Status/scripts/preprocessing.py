import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler


def one_hot_encode_column(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Encodes a categorical column using one-hot encoding.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to encode.

    Returns:
        pd.DataFrame: The encoded DataFrame with new columns for each category.
    """
    encoding = OneHotEncoder(sparse_output=False)

    encoding_data = encoding.fit_transform(data[[column_name]])

    feature_names = encoding.get_feature_names_out([column_name])

    return pd.DataFrame(encoding_data, columns=feature_names, index=data.index)



def one_hot_encode_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes all categorical columns in a DataFrame using one-hot encoding.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The encoded DataFrame with new columns for each category.
    """
    new_df = pd.DataFrame()

    object_feature = (
        data.drop(columns="Loan_Status")
        .select_dtypes(include="object")
        .columns.to_list()
    )

    for col in object_feature:
        new_features = one_hot_encode_column(data, col)

        new_df = pd.concat([new_df, new_features], axis=1)

    return new_df




def ordinal_encoding(data: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Encodes an ordinal column using ordinal encoding.

    Args:
        data (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to encode.

    Returns:
        pd.DataFrame: The encoded DataFrame with the ordinal values.
    """
    encoding = OrdinalEncoder()

    encoding_data = encoding.fit_transform(data[["Risk_Loan"]])

    feature_names = encoding.feature_names_in_

    return pd.DataFrame(encoding_data, columns=feature_names, index=data.index)




def transfer_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    """
    Converts specified columns to their appropriate data types.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the converted data types.
    """
    data["Dependents"] = data["Dependents"].astype("int")

    data["Cluster_Label"] = data["Cluster_Label"].astype("int")

    data["Credit_History"] = data["Credit_History"].astype("int")

    data["ApplicantIncome"] = data["ApplicantIncome"].astype("float")

    data["Risk_Score"] = data["Risk_Score"].astype("float")

    return data




def scale_numerical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scales numerical features using RobustScaler.

    Args:
        data (pd.DataFrame): The input DataFrame containing numerical features.

    Returns:
        pd.DataFrame: The scaled DataFrame with numerical features.
    """
    floating_data = data.select_dtypes(include="float")

    rbs = RobustScaler()

    df_scaling = rbs.fit_transform(floating_data)

    return pd.DataFrame(df_scaling, columns=floating_data.columns)




def preprocess_third_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the third dataset by encoding categorical features, scaling numerical features, and converting data types.

    Args:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    data = transfer_dtypes(data)

    categorical_features = ["Dependents", "Cluster_Label", "Credit_History"]
    numerical_features = data.select_dtypes(include=["number"]).columns.tolist()

    scaled_data = scale_numerical_features(data[numerical_features])

    preprocessed_data = pd.concat([data[categorical_features], scaled_data], axis=1)
    return preprocessed_data
