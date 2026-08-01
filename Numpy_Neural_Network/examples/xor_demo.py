import numpy as np

from nn import Dense, ReLU, Sigmoid, Sequential, MSE


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

model = Sequential([
    Dense(2,4),
    ReLU(),
    Dense(4,1),
    Sigmoid()
])

loss_fn = MSE()

epochs = 5000
lr = 0.1

for epoch in range(epochs):

    prediction = model.forward(X)

    loss = loss_fn.forward(y, prediction)

    gradient = loss_fn.backward()

    model.backward(gradient, lr)

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.6f}")

print("\nFinal Predictions:")
print(model.forward(X))
