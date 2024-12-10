from scripts.config import MODEL_PATH
from scripts.feature import calculate_risk_score
from modeling.model import RandomForest
import streamlit as st
import numpy as np


class LoanApprovalPredictor:
    """
    A class for predicting loan approval based on user input and a loaded machine learning model.
    """


    def __init__(self) -> None:
        """
        Initializes the class with an empty applicant data dictionary and an unloaded model.
        """
        self.applicant_data = {}
        self.loaded_model = None



    @st.cache_resource
    def load_model(_self) -> RandomForest:
        """
        Loads the machine learning model from the specified path.

        Returns:
            The loaded RandomForest model.
        """
        _self.loaded_model = RandomForest.load_model(MODEL_PATH)
        return _self.loaded_model



    def collect_user_input(self) -> dict:
        """
        Collects user input for applicant information and returns a dictionary.

        Returns:
            A dictionary containing applicant data.
        """
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                self.applicant_data["ApplicantIncome"] = st.number_input(
                    "Applicant Income:", min_value=1.0, value=1.0, step=10.0
                )
                self.applicant_data["LoanAmount"] = st.number_input(
                    "Loan Amount:", min_value=1.0, value=1.0, step=10.0
                )
                self.applicant_data["Loan_Amount_Term"] = st.number_input(
                    "Loan Term (Months):", min_value=1.0, value=360.0, step=10.0
                )
                self.applicant_data["Dependents"] = st.selectbox(
                    "Dependents:", options=[0, 1, 2, 3]
                )

            with col2:
                self.applicant_data["CoapplicantIncome"] = st.number_input(
                    "Coapplicant Income:", min_value=1.0, value=1.0, step=10.0
                )
                self.applicant_data["Credit_History"] = st.radio(
                    "Credit History:", options=["Yes", "No"]
                )
                self.applicant_data["Education"] = st.radio(
                    "Education:", options=["Graduate", "Not Graduate"]
                )
                self.applicant_data["Self_Employed"] = st.radio(
                    "Self-Employed:", options=["Yes", "No"]
                )

            st.write("")
            st.write("")

        return self.applicant_data



    def preprocess_data(self) -> tuple[int, float]:
        """
        Preprocesses user data by calculating risk score and assigning risk category.

        Returns:
            A tuple containing the risk category (int) and risk score (float).
        """
        data = self.collect_user_input()
        risk_score = calculate_risk_score(data)

        if risk_score <= 10:
            risk_category = 1
        elif risk_score <= 30:
            risk_category = 2
        else:
            risk_category = 3

        return risk_category, risk_score



    def prepare_final_features(self) -> np.ndarray:
        """
        Prepares final features for the model by encoding categorical data and shaping the array.

        Returns:
            A NumPy array containing features for prediction.
        """
        features = np.zeros(24)
        risk_category, risk_score = self.preprocess_data()

        features[11] = risk_category
        features[14] = self.applicant_data["Credit_History"] == "Yes"
        features[15] = self.applicant_data["ApplicantIncome"]
        features[16] = self.applicant_data["CoapplicantIncome"]
        features[17] = self.applicant_data["LoanAmount"]
        features[18] = self.applicant_data["Loan_Amount_Term"]
        features[23] = risk_score

        return np.reshape(features, (1, -1))



    def run(self) -> None:
        """
        Runs the loan approval prediction process,
        including loading the model, preparing features, and making the prediction.
        """
        self.loaded_model = self.load_model()
        user_features = self.prepare_final_features()

        _, col2, _ = st.columns(3)

        with col2:
            if st.button("Predict Loan Approval", use_container_width=True):
                prediction = self.loaded_model.prediction(user_features)

                if prediction[0] == 1:
                    st.write(
                        "Congratulations! You are likely approved for the loan. 🎉"
                    )
                else:
                    st.write(
                        "Based on your input, loan approval may be less likely. Consider contacting a loan specialist for further guidance. 😔"
                    )




if __name__ == "__main__":
    app = LoanApprovalPredictor()
    app.run()
