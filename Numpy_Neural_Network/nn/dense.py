import numpy as np

class Dense:
    def __init__(self, input_size, output_size):

        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.B = np.zeros((1, output_size))

    def forward(self, X):

        self.X = X
        self.Z = X @ self.W + self.B

        return self.Z

    def backward(self, dZ, lr):

        dW = self.X.T @ dZ
        dB = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T

        self.W -= lr * dW
        self.B -= lr * dB

        return dX
