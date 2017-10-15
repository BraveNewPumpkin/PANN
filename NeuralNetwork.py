import sys
import pandas as pd
import numpy as np
from pathlib import Path
from pprint import pprint

usage = 'NeuralNetwork.py path/to/dataset training_percent maximum_iterations num_hidden_layers num_neurons_per_layer'

if len(sys.argv) < 6:
    print('not enough arguments\n')
    print(usage)
    sys.exit(1)

def main(argv):
    training_dataset_path = Path(argv[1])
    percent_data_for_test = float(argv[2])/100.0
    max_iterations = int(argv[3])
    num_hidden_layers = int(argv[4])
    hidden_layer_sizes = []
    for i in range(5, 4 + num_hidden_layers):
        hidden_layer_sizes.append(int(argv[i]))

    data = pd.read_csv(training_dataset_path)

    (train_data, test_data) = split_train_and_test(data=data, percent_data_for_test=percent_data_for_test)

    classifier_column_names = train_data.columns.str.extract('(classifier=.*)', expand=False).dropna()

    x_train = train_data.drop(classifier_column_names, axis=1)
    y_train = train_data[classifier_column_names]
    x_test = test_data.drop(classifier_column_names, axis=1)
    y_test = test_data[classifier_column_names]

    num_input_nodes = x_train.shape[1]  # number of features in data set
    num_output_nodes = y_train.shape[1]

    neural_network = NeuralNetwork()
    neural_network.train(
        x=x_train,
        y=y_train,
        learning_rate=0.5,
        num_input_nodes=num_input_nodes,
        num_hidden_layers=num_hidden_layers,
        num_output_nodes=num_output_nodes,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iterations=max_iterations
        )
    raw_test_output = neural_network.classify(x_test)
    test_output = pd.DataFrame(data=raw_test_output, columns=classifier_column_names).round()

    #oh how I wish DataFrame.merge would work properly
    #merged = pd.merge(left=y_test, right=test_output, on=classifier_column_names.tolist(), indicator=True)
    #pprint(merged.shape)

    num_correctly_classified = 0
    for row_num in range(0, y_test.shape[0]):
        y_test_row = y_test.iloc[row_num, :]
        test_output_row = test_output.iloc[row_num, :]
        row_matches = None
        for y_test_value, test_output_value in zip(y_test_row, test_output_row):
            if(y_test_value == test_output_value):
                row_matches = True
            else:
                row_matches = False
                break
        if row_matches:
#            print('y_test: ', y_test.iloc[row_num, :].as_matrix(), ' == test_output: ', test_output.iloc[row_num, :].as_matrix())
            num_correctly_classified += 1
    print('error: %f%%' % ((1 - num_correctly_classified / y_test.shape[0]) * 100))


def split_train_and_test(data, percent_data_for_test):
    num_rows_for_test = round(data.shape[0] * percent_data_for_test)
    test_row_indexes = np.random.randint(low=0, high=data.shape[0], size=num_rows_for_test)
    test_data = data.iloc[test_row_indexes, :]
    train_data = data.drop(data.index[test_row_indexes])
    return train_data, test_data


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

    def forward_propagation(self, x):
        hidden_layer_input = np.dot(x, self.hidden_layer_weights)
        hidden_layer_input += self.hidden_layer_bias
        hidden_layer_activations = self.sigmoid(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_activations, self.output_weights)
        output_layer_input += self.output_bias
        output = self.sigmoid(output_layer_input)
        return output, hidden_layer_activations

    def train(self, x, y, learning_rate, num_input_nodes, num_hidden_layers, num_output_nodes, hidden_layer_sizes, max_iterations):
        self.hidden_layer_weights = np.random.uniform(size=(num_input_nodes, num_hidden_layers))
        self.hidden_layer_bias = np.random.uniform(size=(1, num_hidden_layers))
        self.output_weights = np.random.uniform(size=(num_hidden_layers, num_output_nodes))
        self.output_bias = np.random.uniform(size=(1, num_output_nodes))


        for i in range(max_iterations):
            # Forward Propogation
            (output, hidden_layer_activations) = self.forward_propagation(x)

            # Backpropagation
            error = y - output
            slope_output_layer = self.sigmoid_prime(output)
            slope_hidden_layer = self.sigmoid_prime(hidden_layer_activations)
            d_output = error * slope_output_layer
            hidden_layer_error = d_output.dot(self.output_weights.T)
            hidden_layer_delta = hidden_layer_error * slope_hidden_layer
            self.output_weights += hidden_layer_activations.T.dot(d_output) * learning_rate
            self.output_bias += np.sum(d_output, axis=0) * learning_rate
            self.hidden_layer_weights += x.T.dot(hidden_layer_delta) * learning_rate
            self.hidden_layer_bias += np.sum(hidden_layer_delta, axis=0) * learning_rate

            print("Weights for EPOCH: ", i + 1)
            print(self.hidden_layer_weights)

    def classify(self, x):
        (output, hidden_layer_activations) = self.forward_propagation(x)
        return output

main(sys.argv)