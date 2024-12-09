from starlette.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from script.feature import new_features
from script.config import MODEL_PATH
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib


class InputData(BaseModel):
    """
    Data model for input features.

    Attributes:
        f1: Feature 1 (int)
        f2: Feature 2 (int)
        f3: Feature 3 (float)
        f4: Feature 4 (float)
        f5: Feature 5 (float)
        f6: Feature 6 (int)
        f7: Feature 7 (float)
    """

    f1: int
    f2: int
    f3: float
    f4: float
    f5: float
    f6: int
    f7: float


class ModelPredictor:
    """
    Class for making predictions using a trained machine learning model.

    Attributes:
        model: The loaded machine learning model.
        FEATURE_MAPPING: A dictionary mapping feature names to input data attributes.
    """

    FEATURE_MAPPING = {
        "Device Model": "f1",
        "Operating System": "f2",
        "App Usage Time (min/day)": "f3",
        "Screen On Time (hours/day)": "f4",
        "Battery Drain (mAh/day)": "f5",
        "Number of Apps Installed": "f6",
        "Data Usage (MB/day)": "f7",
    }

    def __init__(self, model_path: str) -> None:
        """
        Initializes the ModelPredictor with the path to the saved model.

        Args:
            model_path: Path to the saved model file.
        """
        self.model = joblib.load(model_path)

    def _create_input_dataframe(self, data: InputData) -> pd.DataFrame:
        """
        Creates a pandas DataFrame from the input data.

        Args:
            data: An instance of the InputData class.

        Returns:
            A pandas DataFrame containing the input data.
        """
        return pd.DataFrame(
            {
                col_name: getattr(data, attr_name)
                for col_name, attr_name in self.FEATURE_MAPPING.items()
            },
            index=[0],
        )

    def predict(self, data: InputData) -> List[float]:
        """
        Makes a prediction using the loaded model.

        Args:
            data: An instance of the InputData class.

        Returns:
            A list containing the prediction result.
        """
        base_features = self._create_input_dataframe(data)

        derived_features = pd.DataFrame(new_features(base_features))

        complete_features = pd.concat([base_features, derived_features], axis=1)

        return self.model.predict_client(complete_features).tolist()[0]


app = FastAPI()
predictor = ModelPredictor(MODEL_PATH)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(data: InputData):
    prediction = predictor.predict(data)
    return {"prediction": prediction}
