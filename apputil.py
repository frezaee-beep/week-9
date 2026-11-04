
import pandas as pd
import numpy as np


class GroupEstimate(object):
    """Estimate numerical values based on categorical groups."""
    def __init__(self, estimate):
        """Initialize the model with mean or median estimation."""
        if estimate not in ["mean", "median"]:
            raise ValueError("estimate must be 'mean' or 'median'")

        self.estimate = estimate
    
    def fit(self, X, y):
        """Calculate and store estimates for each group."""
        data = X.copy()
        data["y"] = y

        group_columns = X.columns.tolist()
        self.group_columns = group_columns

        if self.estimate == "mean":
            self.group_estimates = (
                data.groupby(group_columns)["y"].mean()
            )

        elif self.estimate == "median":
            self.group_estimates = (
                data.groupby(group_columns)["y"].median()
          )

        else:
            raise ValueError("estimate must be 'mean' or 'median'")

        return self

    def predict(self, X):
        """Return estimates for new observations."""
        X_new = pd.DataFrame(X, columns=self.group_columns)

        predictions = []

        for _, row in X_new.iterrows():
            key = tuple(row)

            try:
                prediction = self.group_estimates.loc[key]
            except KeyError:
                prediction = np.nan
                print(f"Warning: missing group {key}")

            predictions.append(prediction)

        return np.array(predictions)