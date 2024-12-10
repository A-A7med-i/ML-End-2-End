import random
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


def plot_categorical(data: pd.DataFrame, col: str, custom_colors: list[str]) -> None:
    """
    Creates a dual plot (bar and pie) to visualize the distribution of a categorical variable.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data.
        col (str): The name of the categorical column to visualize.
        custom_colors (list[str]): A list of color codes for the plot elements.
    """
    y = data[col].value_counts()
    x = y.index.tolist()

    num_cat = len(x)

    random_bar_colors = random.sample(custom_colors, num_cat)
    random_pie_colors = random.sample(custom_colors, num_cat)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]])

    # Bar plot
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name="Category Counts",
            marker_color=random_bar_colors,
        ),
        row=1,
        col=1,
    )

    # Pie plot
    fig.add_trace(
        go.Pie(
            labels=x,
            values=y,
            name="Category Percentage",
            hoverinfo="label+percent+value",
            marker=dict(colors=random_pie_colors),
            textinfo="percent+label",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title={
            "text": f"Distribution of {col}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=False,
        width=1000,
        height=600,
    )

    fig.show()


def plot_numerical(data: pd.DataFrame, col: str, custom_colors: list[str]) -> None:
    """
    Creates a dual plot (histogram and box plot) to visualize the distribution of a numerical variable.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data.
        col (str): The name of the numerical column to visualize.
        custom_colors (list[str]): A list of color codes for the plot elements.
    """
    i = random.randint(0, len(custom_colors) - 1)
    j = random.randint(0, len(custom_colors) - 1)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "Histogram"}, {"type": "box"}]],
        subplot_titles=(f"Histogram of {col}", f"Box Plot of {col}"),
    )

    # Histogram
    fig.add_trace(
        go.Histogram(x=data[col], name=f"{col}", marker_color=custom_colors[i]),
        row=1,
        col=1,
    )

    # Box Plot
    fig.add_trace(
        go.Box(y=data[col], name=f"{col}", marker_color=custom_colors[j]), row=1, col=2
    )

    fig.update_layout(
        title={
            "text": f"Distribution of {col}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=False,
        height=600,
        width=1000,
    )

    # Update x-axis and y-axis labels
    fig.update_xaxes(title_text=col, row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_yaxes(title_text=col, row=1, col=2)

    fig.show()


def plot_loan_vs_column(
    data: pd.DataFrame, columns: list[str], custom_colors: list[str]
) -> None:
    """
    Creates a grid of countplots visualizing the relationship between several columns and loan status.

    Args:
        data (pd.DataFrame): Input DataFrame containing the data.
        columns (list[str]): A list of column names to visualize against loan status.
        custom_colors (list[str]): A list of color codes for the plot elements.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 10))

    for i, column in enumerate(columns):
        row = i // 2
        col = i % 2

        sns.countplot(
            data,
            x=column,
            hue="Loan_Status",
            palette={
                "Y": custom_colors[i % len(custom_colors)],
                "N": custom_colors[(i + 1) % len(custom_colors)],
            },
            ax=axes[row, col],
        )
        axes[row, col].set_title(f"{column} vs Loan Status")
        axes[row, col].set_xlabel("")
        axes[row, col].legend(title="Loan Status")

    plt.tight_layout()
    plt.show()
