import numpy as np
import pandas as pd

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.zeros((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

Layer1 = Layer_Dense(784, 16)
Layer2 = Layer_Dense(16, 16)
Layer3 = Layer_Dense(16, 10)

l1_active = Activation_ReLU()
l2_active = Activation_ReLU()
loss_activation = Activation_Softmax()

print("Loading trained weights...")
saved_data = np.load('best_model_weights_2.npz')

Layer1.weights = saved_data['l1_w']
Layer1.biases = saved_data['l1_b']
Layer2.weights = saved_data['l2_w']
Layer2.biases = saved_data['l2_b']
Layer3.weights = saved_data['l3_w']
Layer3.biases = saved_data['l3_b']

# 4. Load Kaggle's test data
print("Loading test.csv...")
test_data = pd.read_csv('test.csv')
X_test = test_data.values / 255.0  

print("Making predictions...")
Layer1.forward(X_test)
l1_active.forward(Layer1.output)

Layer2.forward(l1_active.output)
l2_active.forward(Layer2.output)

Layer3.forward(l2_active.output)
loss_activation.forward(Layer3.output)

predictions = np.argmax(loss_activation.output, axis=1)

print("Formatting submission file...")
submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),  
    'Label': predictions
})

submission.to_csv('submission.csv', index=False)
print("Success! 'submission.csv' is ready to be uploaded to Kaggle.")