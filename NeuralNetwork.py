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

    run_neural_network(X=X,
                       y=y,
                       learning_rate=0.5,
                       num_input_nodes=num_input_nodes,
                       num_hidden_layers=num_hidden_layers,
                       num_output_nodes=num_output_nodes,
                       max_iterations=max_iterations
                       )


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def run_neural_network(X, y, learning_rate, num_input_nodes, num_hidden_layers, num_output_nodes, max_iterations):
    hidden_layer_weights = np.random.uniform(size=(num_input_nodes, num_hidden_layers))
    hidden_layer_bias = np.random.uniform(size=(1, num_hidden_layers))
    output_weights = np.random.uniform(size=(num_hidden_layers, num_output_nodes))
    output_bias = np.random.uniform(size=(1, num_output_nodes))

    for i in range(max_iterations):
        # Forward Propogation
        hidden_layer_input1 = np.dot(X, hidden_layer_weights)
        hidden_layer_input = hidden_layer_input1 + hidden_layer_bias
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1 = np.dot(hiddenlayer_activations, output_weights)
        output_layer_input = output_layer_input1 + output_bias
        output = sigmoid(output_layer_input)

        # Backpropagation
        E = y - output
        slope_output_layer = sigmoid_prime(output)
        slope_hidden_layer = sigmoid_prime(hiddenlayer_activations)
        d_output = E * slope_output_layer
        hidden_layer_error = d_output.dot(output_weights.T)
        hidden_layer_delta = hidden_layer_error * slope_hidden_layer
        output_weights += hiddenlayer_activations.T.dot(d_output) * learning_rate
        output_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
        hidden_layer_weights += X.T.dot(hidden_layer_delta) * learning_rate
        hidden_layer_bias += np.sum(hidden_layer_delta, axis=0, keepdims=True) * learning_rate

        print("Weights for EPOCH: ", i + 1)
        print(hidden_layer_weights)

    print("OUTPUTS")
    print(output)

main(sys.argv)