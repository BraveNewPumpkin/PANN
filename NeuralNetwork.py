import sys
import numpy as np

usage = 'ebp.py dataset_name 200 2 4 2'

if len(sys.argv) < 7:
	print('not enough arguments\n')
	print(usage)
	sys.exit(1)
if len(sys.argv) > 7:
	print('too many arguments\n')
	print(usage)
	sys.exit(1)

training_dataset = argv[1]
training_data_percentage = int(argv[2])
max_iterations = int(argv[3])
no_hidden_layers = int(argv[4])
hidden_layer_sizes = []

import numpy as np


X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

y=np.array([[1],[1],[0]])

def sigmoid (x):
	return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x) * (1 - sigmoid(x))


learning_rate = 0.5
input_neurons = X.shape[1] #number of features in data set

hidden_layer_weights = np.random.uniform(size=(input_neurons,no_hidden_layers))
hidden_layer_bias = np.random.uniform(size=(1,no_hidden_layers))
output_weights = np.random.uniform(size=(no_hidden_layers,output_neurons))
output_bias = np.random.uniform(size=(1,output_neurons))

for i in range(max_iteration):

	#Forward Propogation
	hidden_layer_input1 = np.dot(X,hidden_layer_weights)
	hidden_layer_input = hidden_layer_input1 + hidden_layer_bias
	hiddenlayer_activations = sigmoid(hidden_layer_input)
	output_layer_input1 = np.dot(hiddenlayer_activations,output_weights)
	output_layer_input = output_layer_input1 + output_bias
	output = sigmoid(output_layer_input)

	#Backpropagation
	E = y - output
	slope_output_layer = sigmoid_prime(output)
	slope_hidden_layer = sigmoid_prime(hiddenlayer_activations)
	d_output = E * slope_output_layer
	hidden_layer_error = d_output.dot(output_weights.T)
	hidden_layer_delta = hidden_layer_error * slope_hidden_layer
	output_weights += hiddenlayer_activations.T.dot(d_output) * learning_rate
	output_bias += np.sum(d_output, axis=0,keepdims=True) * learning_rate
	hidden_layer_weights += X.T.dot(hidden_layer_delta) * learning_rate
	hidden_layer_bias += np.sum(hidden_layer_delta, axis=0,keepdims=True) * learning_rate

	print("Weights for EPOCH: ", i+1)
	print(hidden_layer_weights)

print ("OUTPUTS")
print (output)
