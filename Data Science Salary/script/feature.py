import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from .preprocessing import CategoryEncoder

RANDOM_STATE = 0


class ClusteringModel:
    """
    A class for clustering data using KMeans and visualizing the clusters using t-SNE.

    Attributes:
        model: The KMeans model object.
        encoder: The CategoryEncoder object for encoding categorical features.
    """

    def __init__(self, n_clusters=4, n_init="auto", random_state=RANDOM_STATE):
        """
        Initializes the ClusteringModel object.

        Args:
            n_clusters: The number of clusters for KMeans. Defaults to 4.
            n_init: The number of times the K-means algorithm will be run with different centroid seeds.
                   Defaults to "auto".
            random_state: The seed for the random number generator. Defaults to 0.
        """
        self.model = KMeans(
            n_clusters=n_clusters, n_init=n_init, random_state=random_state
        )
        self.encoder = CategoryEncoder()

    def encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes categorical features in the given DataFrame.

        Args:
            data: The DataFrame containing categorical features.

        Returns:
            A new DataFrame with encoded categorical features.
        """
        encoded_data = pd.DataFrame()

        for column in data.select_dtypes(include="object").columns:
            self.encoder.fit(data, column)
            encoded_data[column] = self.encoder.encode(data[column])

        return encoded_data

    def train(self, data: pd.DataFrame) -> None:
        """
        Trains the KMeans model on the given data.

        Args:
            data: The DataFrame containing the features for clustering.
        """
        self.model.fit(data)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predicts the cluster labels for the given data.

        Args:
            data: The DataFrame containing the features for prediction.

        Returns:
            An array containing the predicted cluster labels.
        """
        return self.model.predict(data)

    def build_tsne(self, data: pd.DataFrame) -> np.ndarray:
        """
        Applies t-SNE dimensionality reduction to the given data.

        Args:
            data: The DataFrame containing the features for t-SNE.

        Returns:
            A NumPy array containing the t-SNE transformed features.
        """
        model = TSNE(n_components=2, random_state=RANDOM_STATE)

        new_features = model.fit_transform(data.drop(columns=["Cluster_Label"]))

        return new_features

    def plot_tsne(self, data: pd.DataFrame) -> None:
        """
        Visualizes the clusters using a t-SNE plot.

        Args:
            data: The DataFrame containing the features and cluster labels.
        """
        features = self.build_tsne(data)

        data = go.Scatter(
            x=features[:, 0],
            y=features[:, 1],
            mode="markers",
            marker=dict(
                color=data["Cluster_Label"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Cluster Label"),
            ),
        )

        layout = go.Layout(
            title=dict(text="t-SNE Visualization of Clusters", x=0.5),
            width=1000,
            height=600,
            xaxis=dict(title="t-SNE 1"),
            yaxis=dict(title="t-SNE 2"),
        )

        fig = go.Figure(data=data, layout=layout)

        fig.show()
