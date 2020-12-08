from src.code.models.Model import Model
from sklearn.linear_model import Lasso

class LassoModel(Model):

    def __init__(self, alpha: float):
        self.model = Lasso(alpha=alpha)

    def train(self, X, y):
        self.model.fit(X=X, y=y)

    def predict(self, X):
        return self.model.predict(X)
