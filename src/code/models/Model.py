from abc import ABC, abstractmethod
from sklearn.metrics import r2_score

class Model(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass
