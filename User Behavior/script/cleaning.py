import pandas as pd


DROP_COL = ["User ID"]


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    for column_name in df.columns:
        print(column_name)

        if df[column_name].dtype == "object":
            print(f"Unique values: {', '.join(df[column_name].unique())}")
        else:
            print(f"Number of unique values: {df[column_name].nunique()}")

        print()

    print("Dropping unnecessary columns.....")
    df = df.drop(DROP_COL, axis=1)

    return df
