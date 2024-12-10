import pandas as pd
import joblib


def save_data(data: pd.DataFrame, file_path: str):
    try:
        data.to_csv(file_path, index=False)
        print(f"Saved data to {file_path}")
    except Exception as e:
        print(f"Error: {e}")

def save_model(model, filepath: str):
    try:
        joblib.dump(model, filepath)
        print(f"Model is saved at {filepath}")

    except Exception as e:
        print(f"Error {e}")