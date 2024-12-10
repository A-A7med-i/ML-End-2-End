from flask import Flask, request, jsonify, render_template
from scripts.configuration import MODEL_PATH, ENCODER_PATH
from deployment.schemas.patient import PatientData
from scripts.features import new_features
from typing import List
import pandas as pd
import joblib


class ModelPredictor:
    """
    A class for making predictions using a trained machine learning model.

    Attributes:
        model: The loaded machine learning model.
        encoder: The loaded encoder for preprocessing the input data.
        COLUMNS: A list of categorical columns.
        FEATURES: A list of all features.
        REVERSE_MAP: A dictionary to map predicted labels to their original string representation.
    """

    COLUMNS = [
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Insurance Provider",
        "Admission Type",
        "Medication",
    ]

    FEATURES = [
        "Age",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Date of Admission",
        "Insurance Provider",
        "Billing Amount",
        "Room Number",
        "Admission Type",
        "Discharge Date",
        "Medication",
    ]

    REVERSE_MAP = {0: "Normal", 1: "Inconclusive", 2: "Abnormal"}

    def __init__(self, model_path: str, encoder_path: str):
        """
        Initializes the ModelPredictor with the paths to the model and encoder files.

        Args:
            model_path: Path to the saved model file.
            encoder_path: Path to the saved encoder file.
        """
        self.model = joblib.load(model_path)
        self.encoder = joblib.load(encoder_path)

    def _prepare_features(self, input_data: List) -> pd.DataFrame:
        """
        Prepares the input data for prediction.

        Args:
            input_data: A list of input values.

        Returns:
            A pandas DataFrame containing the prepared features.
        """
        df = pd.DataFrame([dict(zip(self.FEATURES, input_data))])

        df = new_features(df)

        df = df.drop(["Date of Admission", "Discharge Date"], axis=1)

        df[self.COLUMNS] = self.encoder.transform(df[self.COLUMNS])

        return df

    def predict(self, input_data: List) -> str:
        """
        Makes a prediction using the loaded model.

        Args:
            input_data: A list of input values.

        Returns:
            The predicted label.
        """
        features_df = self._prepare_features(input_data)

        prediction = self.model.predict_client(features_df)[0]

        return self.REVERSE_MAP[prediction]


app = Flask(__name__)
model = ModelPredictor(MODEL_PATH, ENCODER_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.get_json()

    patient = PatientData(**form_data)
    processed_data = patient.to_dict()

    prediction = model.predict(list(processed_data.values()))
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True)
