from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import os
import numpy as np
import ipdb

def add_external_data(X):
    X = X.copy()
    df_external = pd.read_csv(
        os.path.join("data","external_features.csv"),
        sep=',',
        low_memory=False
        )
    
    columns_to_use = ['NBPERS19', 'P19_POP1564', 'P19_CHOMEUR1564', 'Q219', 'PPMINI19','GI19']
    X = X.merge(df_external[['CODGEO']+columns_to_use], on='CODGEO',how="left")
    df_external['Departement'] = df_external['CODGEO'].apply(lambda element : element[:2])
    X['Departement'] = X['CODGEO'].apply(lambda element : element[:2])

    missing_columns = (X.isnull().sum()>0)[(X.isnull().sum()>0)].index
    for feature in missing_columns:
        dict_feature = df_external.groupby('Departement')[feature].mean().to_dict()
        X.loc[X[feature].isnull(), feature] = X.loc[X[feature].isnull(), "Departement"].map(dict_feature) 
    return X

def missing_values_department(X, data_location):
    X = X.copy()
    X["Departement"] = X["CODGEO"].apply(lambda element: element[:2])
    data_location["Departement"] = data_location["CODGEO"].apply(lambda element: element[:2])
    data_group = data_location.groupby("Departement")[["latitude", "longitude", "Superficie"]].mean()
    for column in data_group.columns:
        dict_feature = data_group[column].to_dict()
        X.loc[X[column].isnull(), "Departement"].map(
            dict_feature
        ).value_counts()
        X[column].fillna((X[column].mean()), inplace=True)
    X = X.drop(columns=["Departement"])
    return X

def handling_data_Paris(X, data_location):
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

def add_location_data(X):
    X = X.copy()
    df_location = pd.read_csv(
        os.path.join("data","location_codgeo.csv"),
        sep=',',
        low_memory=False
        )
    X = X.merge(df_location, on="CODGEO",how="left")
    X = handling_data_Paris(X, df_location)
    X = missing_values_department(X, df_location)
    return X

def removing_codgeo(X):
    X = X.copy()
    try:
        X = X.drop(columns={"CODGEO", "Departement"})
    except:
        X = X.drop(columns={"CODGEO"})
    return X
    
    
class Regressor(TransformerMixin, BaseEstimator):
    def __init__(self):
        location_data_pipeline = FunctionTransformer(add_location_data, validate=False)
        external_data_pipeline = FunctionTransformer(add_external_data, validate=False)
        removing_codgeo_pipeline = FunctionTransformer(removing_codgeo, validate=False)
        self.regressor = MultiOutputRegressor(Ridge(random_state=57))
        self.model =  Pipeline([
            ("adding_location_data", location_data_pipeline),
            ("adding_insee_data", external_data_pipeline),
            ('remove_codgeo',removing_codgeo_pipeline),
            ("preprocessor", StandardScaler()),
            ("regressor", self.regressor)
        ])
        print(self.model)

    def fit(self, X, y):
        # ipdb.set_trace()
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def get_estimator():
    location_data_pipeline = FunctionTransformer(add_location_data, validate=False)
    external_data_pipeline = FunctionTransformer(add_external_data, validate=False)
    removing_codgeo_pipeline = FunctionTransformer(removing_codgeo, validate=False)
    regressor = MultiOutputRegressor(Ridge(random_state=57))
    final_pipe =  Pipeline([
        ("adding_location_data", location_data_pipeline),
        ("adding_insee_data", external_data_pipeline),
        ('remove_codgeo',removing_codgeo_pipeline),
        ("preprocessor", StandardScaler()),
        ("regressor", regressor)
        ])
    return final_pipe
