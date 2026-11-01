import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')

X = data.drop(columns=['label']).values
X = X / 255

Y = data['label'].values

X_train = X[:28000]
Y_train = Y[:28000]

X_val = X[5600:]
Y_val = Y[5600:]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.sqrt(2 / n_inputs) * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = self.exp_values / np.sum(self.exp_values, axis=1, keepdims=True)

class Loss:
    def calculate(self, outputs, target_outputs):
        sample_losses = self.forward(outputs, target_outputs)
        loss = np.mean(sample_losses)
        return loss

class LossEntropy(Loss):
    def forward(self, y_pred, y_real):
        y_pred = np.array(y_pred)
        y_real = np.array(y_real)

        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if(len(y_real.shape) == 1):
            correction = y_pred_clipped[ range(samples), y_real]
        elif(len(y_real.shape) == 2):
            correction = np.sum(y_pred_clipped * y_real, axis=1)

        neg_log_loss = -np.log(correction)
        return neg_log_loss
    def backward(self, dvalues, y_real):
        samples = len(dvalues)

        if len(y_real.shape) == 2:
            y_real = np.argmax(y_real, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_real] -= 1
        self.dinputs = self.dinputs / samples


class Optimizer_SGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_param(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

Layer1 = Layer_Dense(784, 256)
Layer2 = Layer_Dense(256, 256)
Layer3 = Layer_Dense(256, 10)

l1_active = Activation_ReLU()
l2_active = Activation_ReLU()

loss_activation = Activation_Softmax()
loss_fun = LossEntropy()

optimizer = Optimizer_SGD(learning_rate=0.095)

best_loss = float('inf')
patience = 5
patience_counter = 0

batch_size = 100

epochs = 100

for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        
        X_batch = X_train[i : i + batch_size]
        Y_batch = Y_train[i : i + batch_size]

        Layer1.forward(X_batch)
        l1_active.forward(Layer1.output)

        Layer2.forward(l1_active.output)
        l2_active.forward(Layer2.output)

        Layer3.forward(l2_active.output)

        loss_activation.forward(Layer3.output)

        data_loss = loss_fun.calculate(loss_activation.output, Y_batch)

        predictions = np.argmax(Layer3.output, axis=1)
        accuracy = np.mean(predictions==Y_batch)

        loss_fun.backward(loss_activation.output, Y_batch)

        Layer3.backward(loss_fun.dinputs)

        l2_active.backward(Layer3.dinputs)
        Layer2.backward(l2_active.dinputs)

        l1_active.backward(Layer2.dinputs)
        Layer1.backward(l1_active.dinputs)

        optimizer.update_param(Layer1)
        optimizer.update_param(Layer2)
        optimizer.update_param(Layer3)

    Layer1.forward(X_val)
    l1_active.forward(Layer1.output)
    Layer2.forward(l1_active.output)
    l2_active.forward(Layer2.output)
    Layer3.forward(l2_active.output)

    loss_activation.forward(Layer3.output)
    val_loss = loss_fun.calculate(loss_activation.output, Y_val)

    predictions = np.argmax(Layer3.output, axis=1)
    accuracy = np.mean(predictions==Y_val)

    print(f'Epoch: {epoch} | Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}')

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        
        np.savez('best_model_weights.npz', 
                 l1_w=Layer1.weights, l1_b=Layer1.biases,
                 l2_w=Layer2.weights, l2_b=Layer2.biases,
                 l3_w=Layer3.weights, l3_b=Layer3.biases)
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        break

print('Training Complete')
    


