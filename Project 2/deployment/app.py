import gradio as gr
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from script.configuration import MODEL_PATH, EXPORT_PATH_3


@dataclass
class Config:
    """Configuration for the Salary Predictor application.

    Attributes:
        MODEL_PATH (Path): Path to the saved model file. (default: from script.configuration)
        EXPORT_PATH_3 (Path): Path to the preprocessed data file. (default: from script.configuration)
        DROPDOWN_OPTIONS (dict[str, list[str]]): Dictionary mapping feature names to dropdown menu options.
    """

    MODEL_PATH: Path = Path(MODEL_PATH)
    EXPORT_PATH_3: Path = Path(EXPORT_PATH_3)

    DROPDOWN_OPTIONS = {
        "experience_level": [
            "Senior",
            "Mid/Intermediate level",
            "Entry level",
            "Executive level",
        ],
        "employment_type": ["Full-time", "Part-time", "Contractor", "Freelancer"],
        "employee_residence": [
            "United States",
            "Other",
            "United Kingdom",
            "Canada",
            "Spain",
            "India",
            "Germany",
            "France",
        ],
        "remote_ratio": ["On-Site", "Full-Remote", "Half-Remote"],
        "company_location": [
            "United States",
            "Other",
            "United Kingdom",
            "Canada",
            "Spain",
            "India",
            "Germany",
            "France",
        ],
        "company_size": ["MEDIUM", "LARGE", "SMALL"],
        "job_title": [
            "Data Engineering",
            "Data Science",
            "Other",
            "Machine Learning",
            "Data Architecture",
            "Management",
        ],
    }


class SalaryPredictor:
    """A class for predicting salaries based on user input.

    Attributes:
        config (Config): Configuration object containing paths and options.
        cluster_model (ClusteringModel): Instance of a clustering model (from script.feature).
        encoder (CategoryEncoder): Instance of a category encoder (from script.preprocessing).
        prediction_model (object): Loaded prediction model (loaded from self.config.MODEL_PATH).
        df (pd.DataFrame): DataFrame containing preprocessed data (loaded from self.config.EXPORT_PATH_3).
        encoded_data (pd.DataFrame): Encoded version of self.df using self.cluster_model.encoding.

    Methods:
        _load_model(): Loads the clustering model, category encoder, and prediction model.
        _prepare_data(): Reads the preprocessed data and performs encoding and clustering.
        predict(self, *inputs: str) -> str: Predicts salary based on user input.
            Args:
                *inputs (str): List of user-provided values corresponding to dropdown options.
            Returns:
                str: Formatted string with the predicted salary.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes a SalaryPredictor instance.

        Args:
            config (Config): Configuration object.
        """

        self.config = config
        self._load_model()
        self._prepare_data()

    def _load_model(self) -> None:
        """
        Loads the clustering model, category encoder, and prediction model.

        Raises:
            ImportError: If modules from script.feature or script.preprocessing cannot be imported.
        """

        from script.feature import ClusteringModel
        from script.preprocessing import CategoryEncoder

        self.cluster_model = ClusteringModel()
        self.encoder = CategoryEncoder()
        self.prediction_model = joblib.load(self.config.MODEL_PATH)

    def _prepare_data(self) -> None:
        """
        Reads the preprocessed data, performs encoding and clustering.
        """

        self.df = pd.read_csv(self.config.EXPORT_PATH_3)
        self.encoded_data = self.cluster_model.encoding(self.df)
        self.cluster_model.train(self.encoded_data)

    def predict(self, *inputs: str) -> str:
        """
        Predicts salary based on user input.

        This method takes a variable number of strings as input, representing the user's
        selections from the dropdown menus. It encodes the inputs, predicts the cluster,
        combines the encoded features with the predicted cluster label, and uses the
        combined features to predict the salary using the loaded prediction model.

        Args:
            *inputs (str): List of strings representing user-selected dropdown values.

        Returns:
            str: Formatted string containing the predicted salary.
        """

        input_encoded = []
        for i, inp in enumerate(inputs):
            self.encoder.fit(self.df, self.df.columns[i])
            encoded_value = self.encoder.encode(np.array(inp).reshape(1, -1))
            input_encoded.append(encoded_value)

        input_encoded = np.array(input_encoded).reshape(1, -1)

        cluster_label = self.cluster_model.predict(input_encoded)
        features = np.append(input_encoded, cluster_label).reshape(1, -1)
        salary = self.prediction_model.predict(features)[0]

        return f"Predicted Salary: ${int(salary ** 2):,}"


class SalaryPredictorUI:
    """
    A class for creating a Gradio interface to interact with the SalaryPredictor.

    Attributes:
        predictor (SalaryPredictor): Instance of the SalaryPredictor class.
        config (Config): Configuration object containing dropdown options.

    Methods:
        create_interface(): Creates a Gradio interface with dropdown inputs and a text output.
    """

    def __init__(self, predictor: SalaryPredictor, config: Config) -> None:
        """
        Initializes the SalaryPredictorUI instance.

        Args:
            predictor (SalaryPredictor): Instance of the SalaryPredictor class.
            config (Config): Configuration object containing dropdown options.
        """
        self.predictor = predictor
        self.config = config

    def create_interface(self) -> gr.Interface:
        """
        Creates a Gradio interface for the salary prediction.

        Returns:
            gr.Interface: The created Gradio interface.
        """
        inputs = [
            gr.Dropdown(
                choices=value, label=key.replace("_", " ").title(), interactive=True
            )
            for key, value in self.config.DROPDOWN_OPTIONS.items()
        ]

        return gr.Interface(
            fn=self.predictor.predict,
            inputs=inputs,
            outputs="text",
            title="Salary Predictor",
            theme=gr.themes.Soft(),
        )


def main():
    config = Config()
    predictor = SalaryPredictor(config)
    ui = SalaryPredictorUI(predictor, config)
    ui.create_interface().launch(share=True)


if __name__ == "__main__":
    main()
