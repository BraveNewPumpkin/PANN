import sys
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint

usage = 'NeuralNetwork.py dataset_name 200 2 4 2'

if len(sys.argv) < 6:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)
if len(sys.argv) > 6:
    print('too many arguments\n')
    print(usage)
    sys.exit(1)


def main(argv):
    training_dataset_path = Path(argv[1])
    # training_data_percentage = int(argv[2])
    max_iterations = int(argv[2])
    num_hidden_layers = int(argv[3])
    # hidden_layer_sizes = []

    data = pd.read_csv(training_dataset_path)

    names = data.columns.str.extract('(classifier=.*)', expand=False).dropna()

    X = data.drop(names, axis=1)
    y = data[names]

    num_input_nodes = X.shape[1]  # number of features in data set
    num_output_nodes = y.shape[1]

    neural_netork = NeuralNetwork()
    neural_netork.train(X=X,
                       y=y,
                       learning_rate=0.5,
                       num_input_nodes=num_input_nodes,
                       num_hidden_layers=num_hidden_layers,
                       num_output_nodes=num_output_nodes,
                       max_iterations=max_iterations
                       )



class NeuralNetwork:

    def __init__(self):
        self.hidden_layer_weights = None
        self.hidden_layer_bias = None
        self.output_weights = None
        self.output_bias = None


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forward_propigation(self, X):
        hidden_layer_input = np.dot(X, self.hidden_layer_weights)
        hidden_layer_input = hidden_layer_input + self.hidden_layer_bias
        hidden_layer_activations = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_activations, self.output_weights)
        output_layer_input = output_layer_input + self.output_bias
        output = self.sigmoid(output_layer_input)
        return output, hidden_layer_activations

    def train(self, X, y, learning_rate, num_input_nodes, num_hidden_layers, num_output_nodes, max_iterations):
        self.hidden_layer_weights = np.random.uniform(size=(num_input_nodes, num_hidden_layers))
        self.hidden_layer_bias = np.random.uniform(size=(1, num_hidden_layers))
        self.output_weights = np.random.uniform(size=(num_hidden_layers, num_output_nodes))
        self.output_bias = np.random.uniform(size=(1, num_output_nodes))

        (output, hidden_layer_activations) = self.forward_propigation(X)

        for i in range(max_iterations):
            # Forward Propogation

            # Backpropagation
            E = y - output
            slope_output_layer = self.sigmoid_prime(output)
            slope_hidden_layer = self.sigmoid_prime(hidden_layer_activations)
            d_output = E * slope_output_layer
            hidden_layer_error = d_output.dot(self.output_weights.T)
            hidden_layer_delta = hidden_layer_error * slope_hidden_layer
            self.output_weights += hidden_layer_activations.T.dot(d_output) * learning_rate
            self.output_bias += np.sum(d_output, axis=0) * learning_rate
            self.hidden_layer_weights += X.T.dot(hidden_layer_delta) * learning_rate
            self.hidden_layer_bias += np.sum(hidden_layer_delta, axis=0) * learning_rate

            print("Weights for EPOCH: ", i + 1)
            print(self.hidden_layer_weights)

        print("OUTPUTS")
        print(output)

    def classify(self, X):
        (output, hidden_layer_activations) = self.forward_propigation(X)
        return output

main(sys.argv)