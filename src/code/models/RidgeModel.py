from src.code.models.Model import Model
from sklearn.linear_model import Ridge

class RidgeModel(Model):

    def __init__(self, alpha):
        self.model = Ridge(alpha=alpha)

    def train(self, X, y):
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X)
