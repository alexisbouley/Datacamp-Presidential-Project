import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io

col_target = [
    "% Abs/Ins",
    "% Vot/Ins",
    "% Nuls/Ins",
    "% Blancs/Ins",
    "%_Exp/Ins"
    ]
col_target += [
    "% Autres/Ins",
    "% Voix/Ins_MACRON",
    "% Voix/Ins_MELENCHON",
    "% Voix/Ins_LEPEN"
    ]


def f(x):
    return pd.DataFrame(
        np.average(
            x[col_target],
            weights=x['Inscrits'],
            axis=0
            ),
        index=col_target
    ).T


def g(a):
    return "{0:0=3d}".format(a)


def read_data_2017():

    url = """https://www.data.gouv.fr/fr/datasets/r/f4c23dab-46ff-4799-b217-1ab29db7938b"""

    data_2017 = pd.read_csv(
        url,
        sep=",",
        encoding="latin1",
        engine="python"
    )

    data_2017.dropna(inplace=True)

    data_2017["% Autres/Ins"] = 0

    candidats = ['HAMON', 'ARTHAUD', 'POUTOU', 'CHEMINADE']
    candidats += ['LASSALLE', 'ASSELINEAU', 'FILLON']

    for candidat in candidats:
        data_2017["% Autres/Ins"] += data_2017[candidat + "_ins"]

    data_2017.rename(
        {"MÃLENCHON_ins": "% Voix/Ins_MELENCHON"},
        inplace=True,
        axis=1
        )

    data_2017.rename({"LE PEN_ins": "% Voix/Ins_LEPEN"}, inplace=True, axis=1)
    data_2017.rename({"MACRON_ins": "% Voix/Ins_MACRON"}, inplace=True, axis=1)
    data_2017.rename({"CodeInsee": "CODGEO"}, inplace=True, axis=1)
    data_2017.rename({'Abstentions_ins': "% Abs/Ins"}, inplace=True, axis=1)
    data_2017.rename({'Votants_ins': "% Vot/Ins"}, inplace=True, axis=1)
    data_2017.rename({'Blancs_ins': "% Blancs/Ins"}, inplace=True, axis=1)
    data_2017.rename({'Nuls_ins': "% Nuls/Ins"}, inplace=True, axis=1)
    data_2017.rename({'ExprimÃ©s_ins': "%_Exp/Ins"}, inplace=True, axis=1)

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
        "%_Exp/Vot"
        ]

    candidate_col = [
        "N°Panneau",
        "Sexe",
        "Nom",
        "Prénom",
        "Voix",
        "% Voix/Ins",
        "% Voix/Exp"
        ]

    for candidate in range(1, 13):
        col += [x+"_"+str(candidate) for x in candidate_col]

    data_2022 = pd.read_csv(
        os.path.join(
            "data_source",
            "resultats-par-niveau-burvot-t1-france-entiere.txt"
            ),
        sep=";",
        encoding="latin1",
        engine="python",
        names=col,
        header=0
        )

    col_2022 = ["% Abs/Ins",
                "% Vot/Ins",
                "% Nuls/Ins",
                "% Blancs/Ins",
                "%_Exp/Ins"
                ]

    for candidat in range(1, 13):
        col_2022 += ["% Voix/Ins_" + str(candidat)]

    for col in data_2022[col_2022]:
        data_2022[col] = data_2022[col].str.replace(",", ".").astype('float32')

    data_2022["% Autres/Ins"] = 0

    for candidat in range(1, 13):
        if candidat not in [3, 5, 7]:
            data_2022["% Autres/Ins"] += data_2022["% Voix/Ins_"+str(candidat)]

    data_2022.rename(
            {
                "% Voix/Ins_3": "% Voix/Ins_MACRON",
                "% Voix/Ins_5": "% Voix/Ins_MELENCHON",
                "% Voix/Ins_7": "% Voix/Ins_LEPEN"
            },
            inplace=True,
            axis=1
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
        engine="python"
        )

    url = """https://www.insee.fr/fr/statistiques/fichier/6036907/indic-struct-distrib-revenu-2019-COMMUNES_csv.zip"""
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    data_income = pd.read_csv(
        z.open("FILO2019_DISP_COM.csv"),
        sep=";",
        encoding="latin1",
        engine="python"
        )

    subset_col = ["CODGEO", "NBMEN19", "NBPERS19", "NBUC19", "Q219"]
    data_income = data_income[subset_col]

    data_job.dropna(inplace=True)

    data_features = data_income.merge(data_job, on="CODGEO", suffixes=None)

    return data_features


data_2017 = read_data_2017()
data_2022 = read_data_2022()
data_features = read_data_features()

data_results = data_2017.merge(
    data_2022,
    on="CODGEO",
    suffixes=("_2017", "_2022")
    )

df = data_results.merge(
    data_features,
    on="CODGEO",
    suffixes=None
    )


df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=57)

df_public_train, df_public_test = train_test_split(
    df_train, test_size=0.2, random_state=57)

path = os.path.join('data', 'public')
if not os.path.exists(path):
    os.makedirs(path)

df_train.to_csv(os.path.join('data', 'train.csv'), index=False)
df_test.to_csv(os.path.join('data', 'test.csv'), index=False)


df_public_train.to_csv(
    os.path.join('data', 'public', 'train.csv'),
    index=False
    )

df_public_test.to_csv(
    os.path.join('data', 'public', 'test.csv'),
    index=False
    )
