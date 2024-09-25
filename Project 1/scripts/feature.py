import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go


def new_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates new features from existing ones in the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing loan application data.

    Returns:
        pd.DataFrame: The DataFrame with new features added.
    """
    data["Total_Income"] = data["ApplicantIncome"] + data["CoapplicantIncome"]

    data["Income_Loan_Ratio "] = data["Total_Income"] / data["LoanAmount"]

    data["Loan_Amount_Per_Year"] = data["LoanAmount"] / data["Loan_Amount_Term"]

    data["Debt_To_Income_Ratio"] = data["LoanAmount"] / (data["Total_Income"])

    return data




def dependency_for_risk_score_credit(row: pd.Series) -> int:
    """
    Calculates a dependency score for risk score based on credit history.

    Args:
        row (pd.Series): A single row of data from the DataFrame.

    Returns:
        int: The dependency score for risk score based on credit history.
    """

    if row["Credit_History"] == 1:
        return 0
    else:
        return 30




def dependency_for_risk_score_loan_amount(row: pd.Series) -> int:
    """
    Calculates a dependency score for risk score based on loan amount term.

    Args:
        row (pd.Series): A single row of data from the DataFrame.

    Returns:
        int: The dependency score for risk score based on loan amount term.
    """

    if row["Loan_Amount_Term"] <= 180:
        return 0
    elif row["Loan_Amount_Term"] <= 360:
        return 10
    else:
        return 20




def dependency_for_risk_score_income_ratio(row: pd.Series) -> int:
    """
    Calculates a dependency score for risk score based on income-to-loan ratio.

    Args:
        row (pd.Series): A single row of data from the DataFrame.

    Returns:
        int: The dependency score for risk score based on income-to-loan ratio.
    """

    income_loan_ratio = (row["ApplicantIncome"] + row["CoapplicantIncome"]) / row[
        "LoanAmount"
    ]
    if income_loan_ratio > 3:
        return 0
    elif income_loan_ratio > 2:
        return 10
    elif income_loan_ratio > 1:
        return 20
    else:
        return 30




def dependency_for_risk_score_dependents(row: pd.Series) -> int:
    """
    Calculates a dependency score for risk score based on the number of dependents.

    Args:
        row (pd.Series): A single row of data from the DataFrame.

    Returns:
        int: The dependency score for risk score based on the number of dependents.
    """

    if row["Dependents"] == "3":
        return 15
    elif row["Dependents"] == "2":
        return 10
    elif row["Dependents"] == "1":
        return 5
    else:
        return 0




def dependency_for_risk_score_education(row: pd.Series) -> int:
    """
    Calculates a dependency score for risk score based on education level.

    Args:
        row (pd.Series): A single row of data from the DataFrame.

    Returns:
        int: The dependency score for risk score based on education level.
    """
    if row["Education"] == "Graduate":
        return 0
    else:
        return 10




def dependency_for_risk_score_employed(row: pd.Series) -> int:
    """
    Calculates a dependency score for risk score based on employment status.

    Args:
        row (pd.Series): A single row of data from the DataFrame.

    Returns:
        int: The dependency score for risk score based on employment status.
    """
    if row["Self_Employed"] == "Yes":
        return 10
    else:
        return 0




def calculate_risk_score(row: pd.Series) -> int:
    """
    Calculates the overall risk score for a given row of data.

    Args:
        row (pd.Series): A single row of data from the DataFrame.

    Returns:
        int: The calculated risk score.
    """

    score = 0

    score += dependency_for_risk_score_credit(row)

    score += dependency_for_risk_score_dependents(row)

    score += dependency_for_risk_score_education(row)

    score += dependency_for_risk_score_employed(row)

    score += dependency_for_risk_score_income_ratio(row)

    score += dependency_for_risk_score_loan_amount(row)

    return score




def create_risk_loan_feature(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new feature "Risk_Loan" based on the calculated risk score.

    Args:
        data (pd.DataFrame): The input DataFrame containing loan application data.

    Returns:
        pd.DataFrame: The DataFrame with the new "Risk_Loan" feature added.
    """

    data["Risk_Score"] = data.apply(calculate_risk_score, axis=1)

    data["Risk_Loan"] = pd.cut(
        data["Risk_Score"],
        bins=[-1, 10, 30, 60],
        labels=["Low", "Medium", "High"],
    )

    return data




def encoding(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features and concatenates with numerical features.

    Args:
        data (pd.DataFrame): The input DataFrame containing loan application data.

    Returns:
        pd.DataFrame: The DataFrame with encoded features.
    """
    encoding_feature = pd.get_dummies(
        data[["Self_Employed", "Property_Area", "Risk_Loan"]], dtype=int
    )

    numeric_columns = data.select_dtypes(include=[np.number]).columns

    df_to_clustering = pd.concat([data[numeric_columns], encoding_feature], axis=1)

    return df_to_clustering




def scaling(df_to_scale: pd.DataFrame) -> np.ndarray:
    """
    Scales the numerical features in the DataFrame using RobustScaler.

    Args:
        df_to_scale (pd.DataFrame): The DataFrame containing numerical features to be scaled.

    Returns:
        np.ndarray: The scaled DataFrame as a NumPy array.
    """
    rbs = RobustScaler()

    df_scaling = rbs.fit_transform(df_to_scale)

    return df_scaling




def build_clustering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a clustering model using KMeans and assigns cluster labels to the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing loan application data.

    Returns:
        pd.DataFrame: The DataFrame with the new "Cluster_Label" feature added.
    """
    df_to_clustering = encoding(data)

    df_to_clustering_scaling = scaling(df_to_clustering)

    model = KMeans(n_clusters=5, n_init=50)

    cluster_labels = model.fit_predict(df_to_clustering_scaling)
    data["Cluster_Label"] = cluster_labels

    return data




def get_numerical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a clustering model using KMeans and assigns cluster labels to the DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame containing loan application data.

    Returns:
        pd.DataFrame: The DataFrame with the new "Cluster_Label" feature added.
    """
    numerical_features = (
        data.drop(columns="Cluster_Label")
        .select_dtypes(exclude=["object", "category"])
        .columns
    )

    return numerical_features


def build_pca(data: pd.DataFrame) -> np.ndarray:
    """
    Performs Principal Component Analysis (PCA) on numerical features of the data.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data.

    Returns:
        np.ndarray: A NumPy array containing the transformed data in the reduced dimensionality (2 components by default).
    """
    numerical_features = get_numerical_features(data)

    df_to_scale = data.drop(columns="Cluster_Label")[numerical_features]

    df_scaling = scaling(df_to_scale)

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(df_scaling)

    return pca_result




def visualization_pca(data: pd.DataFrame) -> None:
    """
    Visualizes the PCA results using Plotly.

    Args:
        data (pd.DataFrame): A pandas DataFrame containing the data (assumed to have a "Cluster_Label" column).

    Returns:
        None (displays the visualization and saves an HTML file).
    """

    pca_result = build_pca(data)

    fig = go.Figure(
        data=go.Scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            mode="markers",
            marker=dict(
                color=data["Cluster_Label"],
                colorscale="Viridis",
                size=12,
                opacity=0.8,
                showscale=True,
                colorbar=dict(title="Cluster"),
            ),
            text=data["Cluster_Label"],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title={
            "text": "PCA of Clustered Data",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="First Principal Component",
        yaxis_title="Second Principal Component",
        width=1500,
        height=600,
        coloraxis_colorbar=dict(title="Cluster Label"),
        legend_title="Cluster",
        font=dict(size=12),
        hovermode="closest",
    )

    fig.show()

    fig.write_html(r"F:\End-2-End\Project 1\reports\Figures\pca_visual.html")
