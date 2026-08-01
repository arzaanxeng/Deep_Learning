from .dense import Dense


class Sequential:

    def __init__(self, layers):

        self.layers = layers

    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, gradient, lr):

        for layer in reversed(self.layers):

            if isinstance(layer, Dense):
                gradient = layer.backward(gradient, lr)

            else:
                gradient = layer.backward(gradient)

        return gradient
