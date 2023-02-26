from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import numpy as np


def _add_external_data(X):
    X = X.copy()
    df_external = pd.read_csv(
        os.path.join("data","external_features.csv"),
        sep=',',
        low_memory=False
        )
    X = X.merge(df_external, how='CODGEO')
    return X

def _missing_values_department(X, data_location):
    X = X.copy()
    X["Departement"] = X["CODGEO"].apply(lambda element: element[:2])
    data_location["Departement"] = data_location["CODGEO"].apply(lambda element: element[:2])
    data_group = data_location.groupby("Departement")[["latitude", "longitude", "Superficie"]].mean()
    for column in data_group.columns:
        globals()[f"dict_{column}"] = data_group[column].to_dict()
        X.loc[X[column].isnull(), "Departement"].map(
            globals()[f"dict_{column}"]
        ).value_counts()
        X[column].fillna((X[column].mean()), inplace=True)
    X = X.drop(columns=["Departement"])
    return X

def _handling_data_Paris(X, data_location):
    X = X.copy()
    paris_boroughs = []
    for i in range(1, 21):
        if i < 10:
            paris_boroughs.append(f"7510{i}")
        else:
            paris_boroughs.append(f"751{i}")
    paris_data = data_location[data_location["CODGEO"].isin(paris_boroughs)]
    try:
        X.loc[
            X["CODGEO"].str.contains("75056"), ["latitude", "longitude", "Superficie"]
        ] = paris_data.agg(
            {"latitude": np.mean, "longitude": np.mean, "Superficie": np.sum}
        ).values
    except:
        pass
    return X

def _add_location_data(X):
    X = X.copy()
    df_location = pd.read_csv(
        os.path.join("data","location_codgeo.csv"),
        sep=',',
        low_memory=False
        )
    X = X.merge(df_location, on="CODGEO",how="left")
    X = _handling_data_Paris(X, df_location)
    X = _missing_values_department(X, df_location)
    return X

def _removing_codgeo(X):
    X = X.copy()
    X = X.drop(columns={"CODGEO"})
    return X
    
    
class Regressor(TransformerMixin, BaseEstimator):
    def __init__(self):

        self.regressor = MultiOutputRegressor(Ridge(random_state=57))
        self.preprocessor = StandardScaler()

        self.model = Pipeline([
            ("preprocessor", self.preprocessor),
            ("regressor", self.regressor)
        ])

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)


def get_estimator():
    location_data_pipeline = FunctionTransformer(_add_location_data, validate=False)
    external_data_pipeline = FunctionTransformer(_add_external_data, validate=False)
    regressor = MultiOutputRegressor(Ridge(random_state=57))
    final_pipe = Pipeline([
        ("adding_location_data", location_data_pipeline),
        ("adding_insee_data", external_data_pipeline),
        ("preprocessor", StandardScaler()),
        ("regressor", regressor)
    ])
    reg = Regressor()
    return final_pipe
