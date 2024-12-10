import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go


columns = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Loan_Status",
]


def encode_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns in the provided DataFrame using LabelEncoder.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to be encoded.

    Returns:
        pd.DataFrame: The DataFrame with encoded categorical columns.
    """
    encoder = LabelEncoder()

    for col in columns:
        data[col] = encoder.fit_transform(data[col])

    return data




def train_random_forest_classifier(data: pd.DataFrame) -> pd.Series:
    """
    Performs data selection and feature importance calculation using Random Forest Classifier.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.Series: A Pandas Series containing the feature importances rounded to 3 decimal places.
    """
    data = encode_columns(data)

    data["Risk_Loan"] = data["Risk_Loan"].map({"Low": 0, "Medium": 1, "High": 2})

    X = data.drop(columns=["Loan_Status"])
    y = data["Loan_Status"]

    rfc = RandomForestClassifier()
    rfc.fit(X, y)

    return rfc.feature_importances_.round(3)




def show_feature_importance(X: pd.DataFrame, importance: pd.Series) -> None:
    """
    Visualizes feature importances using Plotly.

    Args:
        X (pd.DataFrame): The DataFrame containing the features.
        importance (pd.Series): A Pandas Series containing the feature importances.
    """
    feature_importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": importance}
    ).sort_values(by="Importance")

    fig = go.Figure(
        go.Bar(
            x=feature_importance["Importance"],
            y=feature_importance["Feature"],
            orientation="h",
            marker=dict(
                color=feature_importance["Importance"],
                colorscale="Viridis",
                showscale=True,
            ),
            showlegend=False,
            opacity=0.8,
            width=0.8,
        )
    )

    fig.update_layout(
        title={
            "text": "Feature Importances",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        xaxis_title="Features",
        yaxis_title="Importance",
        width=1500,
        height=600,
        coloraxis_colorbar=dict(title="Importance"),
    )

    fig.show()

    fig.write_html(r"F:\End-2-End\Project 1\reports\Figures\feature_importances.html")




def correlation_heatmap(data: pd.DataFrame) -> None:
    """
    Generates a correlation heatmap using Plotly.

    Args:
        data (pd.DataFrame): The DataFrame containing the data for calculating correlations.
    """
    corr = data.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Plasma",
            reversescale=True,
            colorbar=dict(title=dict(text="Correlation", side="top")),
        )
    )

    fig.update_layout(
        title={
            "text": "Correlation Heatmap",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        width=1500,
        height=600,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange="reversed",
    )

    fig.show()

    fig.write_html(r"F:\End-2-End\Project 1\reports\Figures\correlation_heatmap.html")
