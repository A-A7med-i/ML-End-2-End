import pickle
import numpy as np
from .baseline import BaseLine
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import roc_curve, roc_auc_score


class RandomForest(BaseLine):
    """
    A Random Forest classification model inheriting from the baseline class.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Initializes the class with training data (X) and labels (y).

        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        super().__init__(X, y)
        self.size = (13, 9)
        self.model = RandomForestClassifier(random_state=self.R_S)
        self.best_model = None



    def cross_validate(self, cv: int = 5) -> tuple[float, float]:
        """
        Performs cross-validation to evaluate model performance.

        Args:
            cv (int, optional): Number of folds for cross-validation. Defaults to 5.

        Returns:
            tuple[float, float]: A tuple containing mean and standard deviation of cross-validation scores.
        """
        scores = cross_val_score(self.model, self.X, self.y, cv=cv)

        print(f"Cross-validation scores: {scores}")
        print(f"Mean CV score: {np.mean(scores):.2f} (+/- {np.std(scores) * 2:.2f})")

        return np.mean(scores), np.std(scores)



    def grid_search(
        self, param_grid: dict
    ) -> tuple[dict, float, RandomForestClassifier]:
        """
        Performs grid search to find optimal hyperparameters for the model.

        Args:
            param_grid (dict): Dictionary containing hyperparameter options for grid search.

        Returns:
            tuple[dict, float, RandomForestClassifier]: A tuple containing the best hyperparameters,
                best score, and best model estimator.
        """
        grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1)

        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_

        return (
            grid_search.best_params_,
            grid_search.best_score_,
            grid_search.best_estimator_,
        )



    def train(self, test_size: float = 0.2) -> None:
        """
        Trains the model using the best model found during grid search (if available).

        Args:
            test_size (float, optional): Split ratio for test set. Defaults to 0.2.
        """
        if self.best_model:
            self.model = self.best_model
        super().train()



    def save_model(self, file_path: str) -> None:
        """
        Saves the trained model to a pickle file.

        Args:
            file_path (str): Path to the file where the model will be saved.
        """
        with open(file_path, "wb") as file:
            pickle.dump(self.model, file)
        print(f"Model saved to {file_path}")



    @classmethod
    def load_model(cls, file_path: str):
        """
        Loads a saved model from a pickle file.

        Args:
            file_path (str): Path to the pickle file containing the model.

        Returns:
            RandomForest: An instance of the RandomForest class with the loaded model.
        """
        with open(file_path, "rb") as file:
            load_model = pickle.load(file)

        instance = cls(None, None)

        instance.model = load_model

        print(f"Model loaded from {file_path}")
        return instance



    def visualization_cm(self) -> None:
        """
        Creates and displays a confusion matrix using Plotly.
        """
        fig = go.Figure(
            go.Heatmap(
                z=self.conf_matrix,
                text=self.conf_matrix,
                hoverinfo="text+x+y",
                colorscale="RdBu",
            )
        )

        fig.update_layout(
            width=1000,
            height=700,
            title={
                "text": "Confusion Matrix",
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            annotations=[
                dict(
                    x=i,
                    y=j,
                    text=str(self.conf_matrix[i][j]),
                    font=dict(size=15),
                    showarrow=False,
                )
                for i in range(len(self.conf_matrix))
                for j in range(len(self.conf_matrix))
            ],
        )

        fig.show()



    def visualization_roc_curve(self) -> None:
        """
        Creates and displays a Receiver Operating Characteristic (ROC) curve using Matplotlib.
        """

        plt.figure(figsize=self.size)

        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred)

        roc_acc = roc_auc_score(self.y_test, self.y_pred)

        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_acc:.02f})",
        )

        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.show()




    def visualization_learning_curve(self) -> None:
        """
        Creates and displays a learning curve using Matplotlib.
        """

        plt.figure(figsize=self.size)

        train_sizes, train_scores, test_scores = learning_curve(
            self.model, self.X, self.y, cv=5
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_score_mean = np.mean(test_scores, axis=1)

        plt.plot(train_sizes, train_scores_mean, lw=2, label="Training score")
        plt.plot(
            train_sizes,
            test_score_mean,
            lw=2,
            linestyle="--",
            marker="o",
            label="Cross-validation score",
        )

        plt.xlabel("Training set size")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend(loc="lower right")
        plt.show()



    def run(self) -> None:
        """
        Runs the entire model training and evaluation process, including visualization of results.
        """
        super().run()
