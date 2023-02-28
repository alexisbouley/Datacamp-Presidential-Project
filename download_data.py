import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io
import ssl
import ipdb

ssl._create_default_https_context = ssl._create_unverified_context

col_target = ["% Abs/Ins", "% Vot/Ins", "% Nuls/Ins", "% Blancs/Ins", "%_Exp/Ins"]
col_target += [
    "% Autres/Ins",
    "% Voix/Ins_MACRON",
    "% Voix/Ins_MELENCHON",
    "% Voix/Ins_LEPEN",
]

departments_to_ignore = ["ZZ", "ZA", "ZN", "ZB", "ZC", "ZD", "ZX", "ZS", "ZW"]


def f(x):
    return pd.DataFrame(
        np.average(x[col_target], weights=x["Inscrits"], axis=0), index=col_target
    ).T


def g(a):
    return "{0:0=3d}".format(a)


def read_data_2017():

    url = """https://www.data.gouv.fr/fr/datasets/r/f4c23dab-46ff-4799-b217-1ab29db7938b"""

    data_2017 = pd.read_csv(url, sep=",", encoding="latin1", engine="python")

    data_2017.dropna(inplace=True)

    data_2017["% Autres/Ins"] = 0

    candidats = ["HAMON", "ARTHAUD", "POUTOU", "CHEMINADE"]
    candidats += ["LASSALLE", "ASSELINEAU", "FILLON"]

    for candidat in candidats:
        data_2017["% Autres/Ins"] += data_2017[candidat + "_ins"]

    data_2017.rename({"MÃLENCHON_ins": "% Voix/Ins_MELENCHON"}, inplace=True, axis=1)

    data_2017.rename({"LE PEN_ins": "% Voix/Ins_LEPEN"}, inplace=True, axis=1)
    data_2017.rename({"MACRON_ins": "% Voix/Ins_MACRON"}, inplace=True, axis=1)
    data_2017.rename({"CodeInsee": "CODGEO"}, inplace=True, axis=1)
    data_2017.rename({"Abstentions_ins": "% Abs/Ins"}, inplace=True, axis=1)
    data_2017.rename({"Votants_ins": "% Vot/Ins"}, inplace=True, axis=1)
    data_2017.rename({"Blancs_ins": "% Blancs/Ins"}, inplace=True, axis=1)
    data_2017.rename({"Nuls_ins": "% Nuls/Ins"}, inplace=True, axis=1)
    data_2017.rename({"ExprimÃ©s_ins": "%_Exp/Ins"}, inplace=True, axis=1)

    data_2017 = data_2017.groupby(by="CODGEO").apply(func=f)
    data_2017 = data_2017.reset_index().drop("level_1", axis=1)

    return data_2017


def read_data_2022():

    col = [
        "Code du département",
        "Libellé du département",
        "Code de la circonscription",
        "Libellé de la circonscription",
        "Code de la commune",
        "Libellé de la commune",
        "Code du b.vote",
        "Inscrits",
        "Abstentions",
        "% Abs/Ins",
        "Votants",
        "% Vot/Ins",
        "Blancs",
        "% Blancs/Ins",
        "% Blancs/Vot",
        "Nuls",
        "% Nuls/Ins",
        "% Nuls/Vot",
        "Exprimésé",
        "%_Exp/Ins",
        "%_Exp/Vot",
    ]

    candidate_col = [
        "N°Panneau",
        "Sexe",
        "Nom",
        "Prénom",
        "Voix",
        "% Voix/Ins",
        "% Voix/Exp",
    ]

    for candidate in range(1, 13):
        col += [x + "_" + str(candidate) for x in candidate_col]

    data_2022 = pd.read_csv(
        os.path.join(
            "data_source", "resultats-par-niveau-burvot-t1-france-entiere.txt"
        ),
        sep=";",
        encoding="latin1",
        engine="python",
        names=col,
        header=0,
    )

    col_2022 = ["% Abs/Ins", "% Vot/Ins", "% Nuls/Ins", "% Blancs/Ins", "%_Exp/Ins"]

    for candidat in range(1, 13):
        col_2022 += ["% Voix/Ins_" + str(candidat)]

    for col in data_2022[col_2022]:
        data_2022[col] = data_2022[col].str.replace(",", ".").astype("float32")

    data_2022["% Autres/Ins"] = 0

    for candidat in range(1, 13):
        if candidat not in [3, 5, 7]:
            data_2022["% Autres/Ins"] += data_2022["% Voix/Ins_" + str(candidat)]

    data_2022.rename(
        {
            "% Voix/Ins_3": "% Voix/Ins_MACRON",
            "% Voix/Ins_5": "% Voix/Ins_LEPEN",
            "% Voix/Ins_7": "% Voix/Ins_MELENCHON",
        },
        inplace=True,
        axis=1,
    )

    series_1 = data_2022["Code du département"]
    series_2 = data_2022["Code de la commune"].apply(g)
    data_2022["CODGEO"] = series_1 + series_2

    data_2022 = data_2022.groupby(by="CODGEO").apply(func=f)
    data_2022 = data_2022.reset_index().drop("level_1", axis=1)

    return data_2022


def read_data_features():

    url = """https://www.insee.fr/fr/statistiques/fichier/6454652/base-ccc-emploi-pop-active-2019.zip"""
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    data_job = pd.read_csv(
        z.open("base-cc-emploi-pop-active-2019.CSV"),
        sep=";",
        encoding="latin1",
        engine="python",
    )

    url = """https://www.insee.fr/fr/statistiques/fichier/6036907/indic-struct-distrib-revenu-2019-COMMUNES_csv.zip"""
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    data_income = pd.read_csv(
        z.open("FILO2019_DISP_COM.csv"), sep=";", encoding="latin1", engine="python"
    )

    subset_col = [
        "CODGEO",
        "NBMEN19",
        "NBPERS19",
        "NBUC19",
        "Q219",
        "PPMINI19",
        "D919",
        "GI19",
        "PPEN19",
    ]
    data_income = data_income[subset_col]
    data_job.dropna(inplace=True)

    data_features = data_income.merge(data_job, on="CODGEO")

    return data_features


def read_geolocation_features():
    url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/correspondance-code-insee-code-postal/exports/csv?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B"
    data_localisation = pd.read_csv(url, sep=";", encoding="latin1", engine="python")
    data_localisation["latitude"] = data_localisation["geo_point_2d"].apply(
        lambda element: float(element.split(",")[0])
    )
    data_localisation["longitude"] = data_localisation["geo_point_2d"].apply(
        lambda element: float(element.split(",")[1])
    )
    data_localisation["CODGEO"] = data_localisation["ï»¿Code INSEE"]
    return data_localisation[["CODGEO", "latitude", "longitude", "Superficie"]]


def filter_final_data(X):
    X = X.copy()
    for problematic_department in departments_to_ignore:
        X = X[~X["CODGEO"].str.contains(problematic_department)]
    return X


def get_external_data(path="."):
    f_name = "external_features.csv"
    data = pd.read_csv(os.path.join(path, "data", f_name), sep=",", low_memory=False)
    return data


def get_location_data(path="."):
    f_name = "location_codgeo.csv"
    data = pd.read_csv(os.path.join(path, "data", f_name), sep=",", low_memory=False)
    return data

def _add_external_data(X):
    X = X.copy()
    df_external = pd.read_csv(
        os.path.join("data","external_features.csv"),
        sep=',',
        low_memory=False
        )
    columns_to_use = ['NBPERS19', 'P19_POP1564', 'P19_CHOMEUR1564', 'Q219', 'PPMINI19','GI19']
    X = X.merge(df_external[['CODGEO']+columns_to_use], how='CODGEO')
    df_external['Departement'] = df_external['CODGEO'].apply(lambda element : element[:2])
    X['Departement'] = X['CODGEO'].apply(lambda element : element[:2])

    missing_columns = (X.isnull().sum()>0)[(X.isnull().sum()>0)].index
    for feature in missing_columns:
        globals()[f"dict_{feature}"] = df_external.groupby('Departement')[feature].mean().to_dict()
        X.loc[X[feature].isnull(), feature] = X.loc[X[feature].isnull(), "Departement"].map(globals()[f"dict_{feature}"]) 
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

def _removing_columns(X, columns):
    X = X.copy()
    X = X.drop(columns=columns)
    return X

def _add_locations(X, data_location):
    X = X.merge(data_location, on='CODGEO', how='left')
    return X

def _delete_missing_locations(X, data_location):
    paris_boroughs = [] 
    for i in range(1,21):
        if i<10:
            paris_boroughs.append(f"7510{i}")
        else:
            paris_boroughs.append(f"751{i}")
    paris_data = data_location[data_location['CODGEO'].isin(paris_boroughs)]
    X.loc[X['CODGEO'].str.contains('75056'),['latitude', 'longitude', 'Superficie'] ]\
    = paris_data.agg({'latitude':np.mean,'longitude':np.mean, "Superficie":np.sum}).values
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

def _add_departement(X):
    new_column = X.loc[:,'CODGEO'].apply(lambda element : element[:2]).values
    X_tr = X.copy()
    X_tr.loc[:, 'Departement'] = new_column
    return X_tr

def preprocessing(X, data_location, df_external_features):
    X = _add_locations(X, data_location)
    X = _delete_missing_locations(X, data_location)
    X = _missing_values_department(X, data_location)

    interesting_external_columns = ["NBPERS19","P19_POP1564","P19_CHOMEUR1564","Q219","PPMINI19","GI19"]
    df_external_analysis = df_external_features[["CODGEO"]+interesting_external_columns]
    
    X = X.merge(df_external_analysis, on='CODGEO',how='left')
    df_external_analysis = _add_departement(df_external_analysis)
    X = _add_departement(X)
    missing_columns = ["PPMINI19","GI19"]
    for feature in missing_columns:
        dict_feature = df_external_analysis.groupby('Departement')[feature].mean().to_dict()
        X_temp = X.copy()
        X_temp.loc[X_temp[feature].isnull(), feature] = X_temp.loc[X_temp[feature].isnull(), "Departement"].map(dict_feature)
        
    return X_temp.fillna(0)


if __name__ == "__main__":
    data_2017 = read_data_2017()
    print("2017 year data has been read")
    data_2022 = read_data_2022()
    print("2022 year data has been read")
    data_features = read_data_features()
    data_location = read_geolocation_features()
    print("External data has been read")
    data_results = data_2017.merge(data_2022, on="CODGEO", suffixes=("_2017", "_2022"))
    df = data_results.copy()
    df = filter_final_data(df)
    
    path = os.path.join("data", "public")
    if not os.path.exists(path):
        os.makedirs(path)
    
    data_features.to_csv(os.path.join("data", "external_features.csv"), index=False)
    data_location.to_csv(os.path.join("data", "location_codgeo.csv"), index=False)
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=57)
    df_train.to_csv(os.path.join("data", "train_raw.csv"), index=False)
    print('Raw data has been saved')
    
    df = preprocessing(df, data_location, data_features)
    df.fillna(0)
    df.to_csv(os.path.join("data", "full_data.csv"), index=True)
    print('Preprocessing has been finished')
    
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=57, shuffle=False)
    df_public_train, df_public_test = train_test_split(
        df_train, test_size=0.2, random_state=57, shuffle=False
    )
    
    #df_train = preprocessing(df_train, data_location, data_features)
    df_train.to_csv(os.path.join("data", "train.csv"), index=False)
    print('Cleaned data has been saved: 25%')
    #df_test = preprocessing(df_test, data_location, data_features)
    df_test.to_csv(os.path.join("data", "test.csv"), index=False)
    print('Cleaned data has been saved: 50%')
    #df_public_train = preprocessing(df_public_train, data_location, data_features)
    df_public_train.to_csv(os.path.join("data", "public", "train.csv"), index=False)
    print('Cleaned data has been saved: 75%')
    #df_public_test = preprocessing(df_public_test, data_location, data_features)
    df_public_test.to_csv(os.path.join("data", "public", "test.csv"), index=False)
    print('Cleaned data has been saved: 100%')
    
