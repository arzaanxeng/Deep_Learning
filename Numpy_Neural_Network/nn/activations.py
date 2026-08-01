import numpy as np


class Sigmoid:

    def forward(self, Z):

        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dA):

        dZ = dA * self.A * (1 - self.A)
        return dZ


class ReLU:

    def forward(self, Z):

        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):

        dZ = dA * (self.Z > 0).astype(float)
        return dZ

# It is more often used as in case of only ReLU for  negative values the gradient always comes out to be zero so the model eventually stops learning !
class LeakyReLU:

    def __init__(self, alpha=0.01):

        self.alpha = alpha

    def forward(self, Z):

        self.Z = Z
        return np.where(Z > 0, Z, self.alpha * Z)

    def backward(self, dA):

        dZ = dA * np.where(self.Z > 0, 1, self.alpha)
        return dZ


class Tanh:

    def forward(self, Z):

        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA):

        dZ = dA * (1 - self.A**2)
        return dZ


class Softmax:

    def forward(self, Z):

        shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp = np.exp(shifted)

        self.A = exp / np.sum(exp, axis=1, keepdims=True)

        return self.A
