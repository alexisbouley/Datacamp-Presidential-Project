import pandas as pd
import os
import numpy as np
import rampwf as rw
from sklearn.model_selection import KFold
from rampwf.score_types.base import BaseScoreType

problem_title = "French Presidential Elections"

_target_column_name = [
    "% Abs/Ins_2022",
    "% Nuls/Ins_2022",
    "% Blancs/Ins_2022",
    "% Autres/Ins_2022",
    "% Voix/Ins_MACRON_2022",
    "% Voix/Ins_MELENCHON_2022",
    "% Voix/Ins_LEPEN_2022"
]

_ignore_column_names = [
    '% Vot/Ins_2022',
    '%_Exp/Ins_2022',
    # 'CODGEO'
]

class MAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='mae', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.mean(abs(y_true - y_pred))
    
class Mixed(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='mixed', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return 0.5 * np.mean(abs(y_true - y_pred)) + 0.5 * np.sqrt(np.mean(np.square(y_true - y_pred)))

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=_target_column_name
    )

# An object implementing the workflow
workflow = rw.workflows.Estimator()

score_types = [
    rw.score_types.RMSE(name="rmse", precision=3),
    MAE(name='mae', precision=3),
    Mixed(name='mixed', precision=3)
]


def get_cv(X, y, random_state=57):

    cv = KFold(n_splits=8)
    return cv.split(X, y)


def _read_data(path, f_name):

    data = pd.read_csv(
        os.path.join(path, 'data', f_name),
        sep=",",
        low_memory=False
        )
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name + _ignore_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)

def get_external_data(path="."):
    f_name = "external_features.csv"
    data = pd.read_csv(
        os.path.join(path, "data",f_name),
        sep=',',
        low_memory=False
    )
    return data