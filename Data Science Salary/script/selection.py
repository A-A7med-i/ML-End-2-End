import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from .preprocessing import CategoryEncoder
from sklearn.ensemble import RandomForestRegressor



def encoding(data: pd.DataFrame):
    encoder = CategoryEncoder()

    encoded_data = pd.DataFrame()

    for column in data.select_dtypes(include='object').columns:
        encoder.fit(data, column)
        encoded_data[column] = encoder.encode(data[column])

    return encoded_data


def build_random_forest(data: pd.DataFrame):
    X = data.drop(columns=["sqrt_salary_in_usd"])
    y = data["sqrt_salary_in_usd"]

    model = RandomForestRegressor(n_estimators=100, max_depth=5)

    model.fit(X, y)

    df = model.feature_importances_.round(3)

    index = data.drop(columns=["sqrt_salary_in_usd"]).columns

    feature_importance_df = pd.DataFrame(df, index=index, columns=["Importance"])

    return feature_importance_df


def plot_importance(data: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=data["Importance"],
            y=data.index,
            orientation="h",
            text=data["Importance"],
            textposition="outside",
            marker_color=px.colors.qualitative.Plotly,
        )
    )

    fig.update_layout(
        title={"text": "Importance Features", "x": 0.5},
        xaxis_title="Feature",
        yaxis_title="Importance",
        width=1000,
        height=700,
    )

    fig.show()


def plot_correlation(data:pd.DataFrame):
    corr = data.corr()

    text = np.round(corr.values, decimals=2)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=corr.columns,
            y=corr.index,
            z=corr.values,
            colorscale="Viridis",
            showscale=True,
            text=text,
            texttemplate="%{text}"
        )
    )

    fig.update_layout(
        title={"text": "Correlation", "x": 0.5},
        xaxis_title="Feature",
        yaxis_title="Correlation",
        width=1000,
        height=700,
    )

    fig.show()
