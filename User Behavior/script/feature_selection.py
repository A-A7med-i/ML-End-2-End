import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

WIDTH: int = 1200
HEIGHT: int = 800


def encoding(data: pd.DataFrame):
    label_encoder = LabelEncoder()

    data["Device Model"] = label_encoder.fit_transform(data["Device Model"])
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    data["Operating System"] = data["Operating System"].map({"Android": 1, "iOS": 0})

    return data


def train_random_forest(X, y):
    rfc = RandomForestClassifier()

    rfc.fit(X, y)

    return rfc.feature_importances_.round(3)


def plot_features_importance(data):
    fig = go.Figure(
        go.Bar(
            x=data["Values"],
            y=data["Features"],
            orientation="h",
            marker=dict(
                color=data["Values"],
                colorscale="Viridis",
                showscale=True,
            )
        )
    )

    fig.update_layout(
        title="Feature Importance",
        title_x=0.5,
        xaxis_title="Feature",
        yaxis_title="Importance",
        width=WIDTH,
        height=HEIGHT,
    )

    fig.show()


def correlation(data: pd.DataFrame):
    corr = data.corr().round(2)

    fig = go.Figure(
        go.Heatmap(
            x=corr.columns,
            y=corr.index,
            z=corr.values,
            text=corr.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorscale='RdBu_r'
        )
    )

    fig.update_layout(
        title=dict(text="Correlation Heatmap", x=0.5),
        width=WIDTH,
        height=HEIGHT
    )

    fig.show()