from src.code.models.Model import Model
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel(Model):

    def __init__(self, n_estimators: int, max_depth: int, min_samples_leaf: int, n_jobs: int, max_features: int):

        self.parameters = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'n_jobs': n_jobs,
            'max_features': max_features,
        }

        self.model = RandomForestRegressor(**self.parameters)

    def train(self, X, y):
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X)
