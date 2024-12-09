import random
import colorcet
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import boxcox
from plotly.subplots import make_subplots


COLORMAP = colorcet.CET_L16
WIDTH, HEIGHT = 1200, 900


def make_subplot(row, col, subplot_titles=None):
    return make_subplots(rows=row, cols=col, subplot_titles=subplot_titles)


def update_layout(fig, title, width=WIDTH, height=HEIGHT, legend=None):
    fig.update_layout(
        title={
            "text": f"{title}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        width=width,
        height=height,
        legend=legend,
    )


def bar_plot(data, column, width, height):
    y = data[column].value_counts()
    x = y.index.to_list()

    color_palette = random.choices(COLORMAP, k=len(COLORMAP))

    data = go.Bar(
        x=x,
        y=y,
        marker=dict(color=color_palette, line=dict(color="black", width=1)),
        hovertext=[f"{val}: {count}" for val, count in zip(x, y)],
        hoverinfo="text",
    )

    fig = go.Figure(data=data)

    update_layout(fig, column, width, height)

    fig.update_traces(texttemplate="%{y}", textposition="outside")

    fig.show()


def salary_plot(data, column, width, height):
    fig = make_subplot(1, 2)

    fig.add_trace(
        go.Histogram(
            x=data[column], name="Original", marker_color=random.choice(COLORMAP)
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=np.log(data[column]),
            name="Log-transformed",
            marker_color=random.choice(COLORMAP),
        ),
        row=1,
        col=2,
    )

    update_layout(fig, column, width, height)

    fig.update_xaxes(title_text=f"{column}", row=1, col=1)
    fig.update_xaxes(title_text=f"Log {column}", row=1, col=1)

    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    mean_salary = data[column].mean()
    mean_salary_log = np.log(data[column]).mean()

    fig.add_vline(
        x=mean_salary,
        line_dash="dash",
        line_color=random.choice(COLORMAP),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=mean_salary, y=1, text="Mean", showarrow=True, arrowhead=2, row=1, col=1
    )

    fig.add_vline(
        x=mean_salary_log,
        line_dash="dash",
        line_color=random.choice(COLORMAP),
        row=1,
        col=2,
    )
    fig.add_annotation(
        x=mean_salary_log, y=1, text="Mean", showarrow=True, arrowhead=2, row=1, col=2
    )

    fig.show()


def group_salary(data, column):
    if data[column].nunique() <= 5:
        new_df = (
            data.groupby(by=[column], as_index=False)["salary_in_usd"]
            .mean()
            .sort_values(by="salary_in_usd", ascending=False)
        )

    else:
        new_df = (
            data.groupby(by=[column], as_index=False)["salary_in_usd"]
            .mean()
            .sort_values(by="salary_in_usd", ascending=False)
            .round(3)[:5]
        )

    return new_df


def plot_grouping_salary(df, column, width, height):
    data = []

    new_df = group_salary(df, column)

    legend = dict(
        title=column,
        orientation="h",
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=1,
    )

    for category, group_df in new_df.groupby(column):
        trace = go.Line(
            x=group_df[column],
            y=group_df["salary_in_usd"],
            name=category,
            marker=dict(color=random.choice(COLORMAP), size=8, line=dict(width=2)),
            line=dict(color=random.choice(COLORMAP)),
        )

        data.append(trace)

    fig = go.Figure(data=data)

    update_layout(fig, column, width, height, legend=legend)

    fig.show()


def top_5_salary_each_year(data):
    top_5_data = (
        data.groupby(by=["work_year", "job_title"], as_index=False)["salary_in_usd"]
        .mean()
        .round(3)
        .sort_values(by=["work_year", "salary_in_usd"], ascending=[True, False])
        .groupby(by="work_year")
        .head(5)
    )

    return top_5_data


def plot_top_5_salaries(data, width, height):

    top_5_data = top_5_salary_each_year(data)

    years = top_5_data["work_year"].unique()

    subplot_titles = [f"Top 5 Job Titles by Average Salary in {year}" for year in years]

    fig = make_subplot(len(years), 1, subplot_titles)

    for i, year in enumerate(years, start=1):
        year_data = top_5_data[top_5_data["work_year"] == year].sort_values(
            by="salary_in_usd", ascending=True
        )

        fig.add_trace(
            go.Bar(
                x=year_data["salary_in_usd"],
                y=year_data["job_title"],
                orientation="h",
                text=year_data["salary_in_usd"].apply(lambda x: f"${x:,.0f}"),
                textposition="outside",
                marker_color=px.colors.qualitative.Set3,
                name=str(year),
            ),
            row=i,
            col=1,
        )

        fig.update_xaxes(title=f"Year {year}", row=i, col=1)

    title = "Top 5 Job Titles by Average Salary per Year"

    update_layout(fig, title, width, height)

    fig.show()


def plot_salary_transformations(df, width, height):
    title = "Distribution Of Salary With Transformations"

    legend = dict(
        orientation="v",
        yanchor="bottom",
        y=1,
        xanchor="right",
        x=1,
    )

    tr, _ = boxcox(df["salary_in_usd"])

    fig = make_subplot(2, 2)

    fig.add_trace(
        go.Box(
            x=df["salary_in_usd"], name="Salary", line_color=random.choice(COLORMAP)
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Box(
            x=np.log(df["salary_in_usd"]),
            name="Log Salary",
            line_color=random.choice(COLORMAP),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Box(
            x=np.sqrt(df["salary_in_usd"]),
            name="Sqrt Salary",
            line_color=random.choice(COLORMAP),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Box(x=tr, name="Box-Cox Salary", line_color=random.choice(COLORMAP)),
        row=2,
        col=2,
    )

    update_layout(fig, title, width, height, legend=legend)

    fig.update_xaxes(title_text="Salary (USD)", row=1, col=1)
    fig.update_xaxes(title_text="Log Salary", row=1, col=2)
    fig.update_xaxes(title_text="Square Root Salary", row=2, col=1)
    fig.update_xaxes(title_text="Box-Cox Transformed Salary", row=2, col=2)

    fig.show()
