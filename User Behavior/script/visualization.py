import random
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


WIDTH: int = 1200
HEIGHT: int = 800
FIGSIZE: tuple = (15, 12)
STYLE: str = "whitegrid"


def random_colors(counts):
    return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(counts)]


def category_distribution(data: pd.DataFrame, category_name: str):
    counts = data[category_name].value_counts()

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])

    bar_colors = random_colors(len(counts))

    pie_colors = random_colors(len(counts))

    fig.add_trace(
        go.Bar(x=counts.index, y=counts.values, marker_color=bar_colors), row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hoverinfo="label+percent+value",
            textinfo="percent+label",
            marker=dict(colors=pie_colors),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Distribution of {category_name}",
        title_x=0.5,
        showlegend=False,
        width=WIDTH,
        height=HEIGHT,
    )

    fig.show()


def numerical_distribution(data: pd.DataFrame, column_name: str):
    colors = random_colors(10)

    color = random.choice(colors)

    data = go.Histogram(
        x=data[column_name], nbinsx=200, marker=dict(color=color, opacity=0.8)
    )

    layout = go.Layout(
        title=f"Histogram of {column_name}",
        title_x=0.5,
        showlegend=False,
        width=WIDTH,
        height=HEIGHT,
    )

    fig = go.Figure(data=[data], layout=layout)

    fig.show()


def box_plot(data: pd.DataFrame, numerical_columns):
    n_plots = len(numerical_columns)

    colors = random_colors(n_plots)

    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGSIZE)

    axes_flat = axes.ravel()

    for i, col in enumerate(numerical_columns):
        sns.boxplot(data=data, x=col, ax=axes_flat[i], color=colors[i])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.show()


def scatter_plot(data: pd.DataFrame, x_metric, y_metric):
    fig = px.scatter(
        x=data[x_metric],
        y=data[y_metric],
        color=data["User Behavior Class"],
        width=WIDTH,
        height=HEIGHT,
    )

    fig.update_layout(
        template="plotly_white",
        legend_title_text="User Behavior",
        xaxis_title=x_metric,
        yaxis_title=y_metric,
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
    )

    fig.update_traces(
        marker=dict(size=10),
    )

    fig.show()


def create_count_plot(data: pd.DataFrame, category_column: str, value_column: str):
    colors = random_colors(len(data[category_column].unique()))

    fig = px.histogram(
        x=data[category_column],
        color=data[value_column],
        barmode="group",
        text_auto=True,
        width=WIDTH,
        height=HEIGHT,
        color_discrete_sequence=colors,
    )

    fig.update_layout(
        title=f"{category_column}",
        title_x=0.5,
        xaxis_title=category_column,
        yaxis_title="Count",
        showlegend=True,
        bargap=0.2,
    )

    fig.update_traces(
        opacity=0.8,
        textposition="auto",
    )

    fig.show()


def grouping_plot(data: pd.DataFrame, category_column: str):
    fig = px.scatter(
        data_frame=data,
        x=category_column,
        y="Number of Apps Installed",
        title=f"Averages by {category_column}",
        size="Battery Drain (mAh/day)",
        color="Age",
        width=WIDTH,
        height=HEIGHT,
    )

    fig.update_layout(
        title_x=0.5,
        xaxis_title=f"{category_column}",
        yaxis_title="Averages",
        showlegend=True,
    )

    fig.show()


def line_plot(data: pd.DataFrame, x_column: str, y_column: str):
    sns.set_style(STYLE)

    plt.figure(figsize=FIGSIZE)

    colors = random_colors(20)

    color = random.choice(colors)

    sns.lineplot(
        x=data[x_column],
        y=data[y_column],
        markers="o",
        color=color,
        linewidth=2,
        markersize=8,
    )

    plt.title(
        f"{x_column} vs Average of {y_column}", pad=20, fontsize=12, fontweight="bold"
    )

    plt.xlabel(f"{x_column}", fontsize=10)
    plt.ylabel(f"{y_column}", fontsize=10)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
