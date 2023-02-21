from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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

    reg = Regressor()
    return reg.model
