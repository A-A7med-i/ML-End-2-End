from plotly.express.colors import qualitative
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import random

COLORS = qualitative.Set3 + qualitative.Set1 + qualitative.Vivid

WIDTH = 1200
HEIGHT = 800


def box_plot(data: pd.DataFrame, column: str):
    data = go.Box(
        x=data[column],
        boxpoints="outliers",
        marker_color=random.choice(COLORS),
        boxmean=True,
    )

    layout = go.Layout(
        title={
            "text": f"{column} Distribution Box Plot",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24},
        },
        xaxis_title="Age",
        yaxis_title="Distribution",
        showlegend=True,
        plot_bgcolor="white",
        width=WIDTH,
        height=HEIGHT,
    )

    fig = go.Figure(data=data, layout=layout)

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    fig.show()


def bar_plot(data: pd.DataFrame, column: str):
    counts = data[column].value_counts()

    data = go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=COLORS,
        opacity=0.8,
        name="Distribution",
    )

    layout = go.Layout(
        title={
            "text": f"{column} Distribution",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24},
        },
        template="plotly_white",
        xaxis_title="Categories",
        yaxis_title="Values",
        bargap=0.2,
        showlegend=True,
        width=WIDTH,
        height=HEIGHT,
    )

    fig = go.Figure(data=data, layout=layout)

    fig.update_traces(texttemplate="%{y: .2f}", textposition="outside")

    fig.show()


def plot_average_billing_scatter(df: pd.DataFrame, category_column: str):
    billing_averages = (
        df.groupby(category_column)["Billing Amount"].mean().round(3).reset_index()
    )

    fig = px.scatter(
        billing_averages,
        x=category_column,
        y="Billing Amount",
        color="Billing Amount",
        size="Billing Amount",
        template="plotly_white",
        width=WIDTH,
        height=HEIGHT,
    )

    fig.update_layout(
        title={
            "text": f"Average Billing by {category_column}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 24},
        },
        xaxis_title=category_column,
        yaxis_title="Average Billing Amount",
        showlegend=True,
    )

    fig.show()
