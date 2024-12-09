import joblib
import pandas as pd


def save_data(data: pd.DataFrame, path_file: str) -> None:
    try:
        data.to_csv(path_file, index=False)
        print(f"Data saved successfully at {path_file}")
    except Exception as e:
        print(f"Error: {e}")


def save_model(model, path):
    try:
        joblib.dump(model, path)
        print(f"Model saved successfully to {path}")
    except Exception as e:
        print(f"Error saving model: {e}")
