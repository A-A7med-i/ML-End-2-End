import pandas as pd


def new_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["Length Of Stay"] = (
        pd.to_datetime(df["Discharge Date"]) - pd.to_datetime(df["Date of Admission"])
    ).dt.days
    df["Age Group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 50, 65, 100],
        labels=[0, 1, 2, 3, 4],
    )
    df["Admission Season"] = pd.to_datetime(df["Date of Admission"]).dt.month.map(
        {
            12: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 2,
            9: 3,
            10: 3,
            11: 3,
        }
    )
    df["Cost Per Day"] = (df["Billing Amount"] / df["Length Of Stay"]).round(3)

    return df
