import numpy as np

#This project is created by Alper Canberk

"""
TODO:

1.ADD BIAS
2.FIX THE ERROR WITH MULTIPLE OUTPUT NODES
3).MAKE AN OPTION WHERE THE PROGRAM SAVES THE WEIGHTS, ERRORS AND THE LAST LAYER VALUES
TO A FILE AND CAN USE IT LATER WHEN THERE'S MORE DATA TO BACK PROPAGATE

"""


class NeuralNetwork:
	def __init__(self, nodes, learning_rate):

		self.layer_count = len(nodes)

		self.weights = []
		self.biases = [] 
		self.errors = [0.0]*self.layer_count  
		self.layer_vals = [0.0]*self.layer_count  

		self.lr = learning_rate


		# This loop sets the initial weights as random values.
		for layer_index in range(0, len(nodes)-1):
			node_count = nodes[layer_index]
			next_node_count = nodes[layer_index+1]
			self.weights.append(np.matrix(np.random.rand(next_node_count, node_count)))
			# For each weight matrix, the column number is the node count of the next layer.
			self.biases.append(np.random.randn())

	def activation_function(self, x):
		return np.divide(1.0, (1 + np.exp(-x)))
		# this program uses the sigmoid as the activation function.

	def activation_function_derivative(self, x):
		return np.multiply(x ,(1 - x))
		#	This is a fake derivative return function, it doesn't
		# actually return the derivative of the activation_function.
		# It only works because the input is already sigmoided and the 
		# derivative of sigmoid is sigmoid(x)*(1-sigmoid(x)).
		# I will definitely fix this later.

	def feedforward(self, input_array):
		self.layer_vals[0] = np.matrix(input_array).T
		for weights_i in range(0, len(self.weights)):

			weights = self.weights[weights_i]
			bias = self.biases[weights_i]
			values = self.layer_vals[weights_i]
			
			#Y = activationFunction(W * I + B)

			# The user input is processed for each layer and passed on
			# to the next layer.
			y = self.activation_function(np.dot(weights, values))

			# We need to save the outputs as an array to be able to use it
			# for back propagation.
			self.layer_vals[weights_i + 1] = y

	  
		# Return the output of this process.
		return self.layer_vals[-1]

	def backprop(self, inputs, targets):
		# It makes a guess.
		self.feedforward(inputs)
		# The target inputs array is turned into a matrix.
		targets = np.matrix(targets).T
		# Output errors are calculated.
		output_errors = np.subtract(targets, self.layer_vals[-1])

		self.errors[-1] = output_errors

		# Looping backwards through the weights to adjust each one.
		for weight_index in range(len(self.weights) - 1, -1, -1):

			# Distribute the error that ocurred in the next layer based on
			# their weights which in this case can be thought as responsibilities
			# for the error.
			
			weights_T = self.weights[weight_index].T
			errors = np.dot(weights_T, self.errors[weight_index + 1])
			self.errors[weight_index] = errors

			#Sorry for the ugly code, np matrix multiplication functions are static.
			# gradient = 
			# learning_rate * errors_right * d(activationFunction(outputs_right))/d(outputs_right) * outputs_left
			gradients = self.activation_function_derivative(self.layer_vals[weight_index + 1])
			gradients = np.multiply(gradients, self.errors[weight_index + 1])
			gradients = np.multiply(gradients, self.lr)

			deltas = np.dot(gradients, self.layer_vals[weight_index].T)

			# Adjust the weights with the new deltas.
			self.weights[weight_index] = np.add(self.weights[weight_index], deltas)

	def train(self, iterations, inputs, answers, pattern_description=""):
		#A nice training function with outputs.
		if(pattern_description != ""):
			print "-------------------"
			print pattern_description
			print "-------------------"

		for i in range(iterations):
			index = np.random.randint(0,len(inputs))
			key = inputs[index]
			value = answers[index]
			nn.backprop(key, value)

		for i in range(0, len(inputs)):
			output = nn.feedforward(inputs[i])
			print "input:"+str(inputs[i])+" expected:"+ str(answers[i])+" output "+str(i)+":" + str(output)


if __name__ == "__main__":
	
	y = np.array([[0],[1],[1],[0]])
	nn = NeuralNetwork([3,5,1], 1)
	
	#Example: the answer is 1 if the 1st and the 3rd list elements are 1.

	inputs = [
	[0,0,0],#0
	[0,0,1],#0
	[0,1,0],#0
	[0,1,1],#0
	[1,0,0],#0
	[1,0,1],#1
	[1,1,0],#0
	[1,1,1] #1

	]
	answers = [
	[0],
	[0],
	[0],
	[0],
	[0],
	[1],
	[0],
	[1]
	]

	nn.train(10000, inputs, answers, "the answer is 1 if the 1st and the 3rd list elements are 1.")




