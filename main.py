import numpy as np

class NeuralNetwork:
    def __init__(self, nodes):
        self.weights = []
        self.biases = []           
        self.lr = 0.1

        for layer_index in range(0, len(nodes)-1):
            node_count = nodes[layer_index]
            next_node_count = nodes[layer_index+1]
            self.weights.append(np.matrix(np.random.rand(next_node_count, node_count)))
            self.biases.append(np.random.randn())
        print "\narray of weights:"
        for i in range(0, len(self.weights)):
            print("weights "+str(i)+":")
            print(self.weights[i])
        print "---------\n" 

    def activation_function(x):
        return 1.0/(1+ np.exp(-x))

    def activation_function_derivative(x):
        return x * (1.0 - x)
        
    def feedforward(self, input_array):
        layer_vals = [np.matrix(input_array).T]
        print "input:"
        print layer_vals[0]
        print ""
        for weights_i in range(0, len(self.weights)):

            weights = self.weights[weights_i]
            bias = self.biases[weights_i]
            values = layer_vals[weights_i]
            
            #Y = activationFunction(W * I + B)
            y = self.activation_function(np.add(np.dot(weights, values), bias))

            print str(weights_i) + ":"
            print "weights:"
            print weights
            print "bias:"
            print bias
            print "values:"
            print values
            print "result:"

            print y

            layer_vals.append(y)

        print "feedforward:\n"
        for i in range(0, len(layer_vals)):
            print "layer vals "+str(i)+":"
            print layer_vals[i]

        return layer_vals[-1]

    def backprop(self, inputs, targets):
        print("BACKPROP STARTING")
        outputs = self.feedforward(inputs)
        targets = np.matrix(targets)

        output_errors = np.subtract(targets, outputs)

        print "BACKPROP!"
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        
        # update the weights with the derivative (slope) of the loss function
        print output_errors




if __name__ == "__main__":
    X = np.array([[0,0,0],
                  [0,1,0],
                  [1,0,0],
                  [1,1,0]])
    y = np.array([[0],[1],[1],[0]])
    nn = NeuralNetwork([3,2,1])
    
    print(nn.feedforward([0,1,0]))
    nn.backprop(
        [[1, 0, 1],
        [0, 0, 1]],
        [[1,0]]
        )


