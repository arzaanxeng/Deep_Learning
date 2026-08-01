import numpy as np


class MSE:

    def forward(self, y_true, y_pred):

        self.y_true = y_true
        self.y_pred = y_pred

        return np.mean((y_true - y_pred) ** 2)

    def backward(self):

        n = self.y_true.shape[0]

        dA = (2 / n) * (self.y_pred - self.y_true)

        return dA
