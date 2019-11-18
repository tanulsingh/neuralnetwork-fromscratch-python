import numpy as np
import dill

class neural_network:
    '''
    This is a custom neural netwok package built from scratch with numpy.
    This code is not optimized and should not be used with real-world examples.
    It's written for learning purposes only.
    The Neural Network as well as its parameters and training method and procedure will 
    reside in this class.

    Examples
    ---
    >>> import neuralnet as nn
    >>> net = nn.neural_network(3, [784, 20, 10], [None, "tanh", "softmax"], cost_function="cross_entropy")

    This means :
    1 hidden layer with 20 neurons
    1 output layer with 10 neurons
    
    '''


    def __init__(self, num_layers, num_nodes, activation_function, cost_function):
        '''
        
        It takes 4 things as inputs:
        1. num_layers: number of layers in the network.
        2. num_nodes: It is a list of size num_layers, specifying the number of nodes in each layer.
        3. activation_function: It is also a list, specifying, the activation function for each layer 
        (activation function for first layer will usually be None. It can take values sigmoid, tanh, relu, softmax.)
        4. cost_function: Function to calculate error between predicted output and actual label/target. 
        It can take values mean_squared, cross_entropy.
        — Layers are initialized with the given number of nodes in each layer. Weights associated with every layer are initialized.

       '''
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.layers = []
        self.cost_function = cost_function

        if not num_layers == len(num_nodes):
            raise ValueError("Number of layers must match number node counts")

        for i in range(num_layers):
            if i != num_layers-1:
                layer_i = layer(num_nodes[i], num_nodes[i+1], activation_function[i])
            else:
                layer_i = layer(num_nodes[i], 0, activation_function[i])
            self.layers.append(layer_i)

    def check_training_data(self, batch_size, inputs, labels):
        self.batch_size = batch_size
        if not len(inputs) % self.batch_size == 0:
            raise ValueError("Batch size must be multiple of number of inputs")
        if not len(inputs) == len(labels):
            raise ValueError("Number of inputs must match number of labels")
        for i in range(len(inputs)):
            if not len(inputs[i]) == self.num_nodes[0]:
                raise ValueError("Length of each input data must match number of input nodes")
            if not len(labels[i]) == self.num_nodes[-1]:
                raise ValueError("Length of each label data must match number of output nodes")

    def train(self, batch_size, inputs, labels, num_epochs, learning_rate, filename):
        '''

         It takes 6 inputs:
        1. batch_size: Mini batch size for gradient descent.
        2. inputs: Inputs to be given to the network.
        3. labels: Target values.
        4. num_epochs: Number of epochs i.e. how many times the program should iterate over all training .
        5. learning_rate: Learning rate for the algorithm, as discussed in DL01.
        6. filename: The name of the file that will finally store all variables after training. (filename must have the extension .pkl).

        '''
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.check_training_data(self.batch_size, inputs, labels)
        for j in range(num_epochs):
            i = 0
            print("== EPOCH: ", j+1, "/", num_epochs, " ==")
            while i+batch_size != len(inputs):
                print("Training with ", i+batch_size+1, "/", len(inputs), end="\r")
                self.error = 0
                self.forward_pass(inputs[i:i+batch_size])
                self.calculate_error(labels[i:i+batch_size])
                self.back_pass(labels[i:i+batch_size])
                i += batch_size
            self.error /= batch_size
            print("\nError: ", self.error)
        print("Saving...")
        dill.dump_session(filename)

    def forward_pass(self, inputs):
        '''

         It takes just 1 input:
         1. inputs: Mini batch of inputs.
         This function multiplies inputs with weights, applies the activation function and stores the output as activations of next layer.
         This process is repeated for all layers, until we have some activations in the output layer.

         '''
        self.layers[0].activations = inputs
        for i in range(self.num_layers-1):
            temp = np.add(np.matmul(self.layers[i].activations, self.layers[i].weights_for_layer), self.layers[i].bias_for_layer)
            if self.layers[i+1].activation_function == "sigmoid":
                self.layers[i+1].activations = self.sigmoid(temp)
            elif self.layers[i+1].activation_function == "softmax":
                self.layers[i+1].activations = self.softmax(temp)
            elif self.layers[i+1].activation_function == "relu":
                self.layers[i+1].activations = self.relu(temp)
            elif self.layers[i+1].activation_function == "tanh":
                self.layers[i+1].activations = self.tanh(temp)
            else:
                self.layers[i+1].activations = temp

    def relu(self, layer):
        layer[layer < 0] = 0
        return layer

    def softmax(self, layer):
        exp = np.exp(layer)
        if isinstance(layer[0], np.ndarray):
            return exp/np.sum(exp, axis=1, keepdims=True)
        else:
            return exp/np.sum(exp, keepdims=True)

    def sigmoid(self, layer):
        return np.divide(1, np.add(1, np.exp(np.negative(layer))))

    def tanh(self, layer):
        return np.tanh(layer)

    def calculate_error(self, labels):
        if len(labels[0]) != self.layers[self.num_layers-1].num_nodes_in_layer:
            print ("Error: Label is not of the same shape as output layer.")
            print("Label: ", len(labels), " : ", len(labels[0]))
            print("Out: ", len(self.layers[self.num_layers-1].activations), " : ", len(self.layers[self.num_layers-1].activations[0]))
            return

        if self.cost_function == "mean_squared":
            self.error += np.mean(np.divide(np.square(np.subtract(labels, self.layers[self.num_layers-1].activations)), 2))
        elif self.cost_function == "cross_entropy":
            self.error += np.negative(np.sum(np.multiply(labels, np.log(self.layers[self.num_layers-1].activations))))

    def back_pass(self, labels):
        '''
        It takes 1 input:
        1. labels: Mini batch of labels.
        This function implements the backpropagation algorithm.
        Basically what it does is, it calculates the gradient, multiplies it with a learning rate and subtracts the product from the existing weights. 
        This is done for all layers, from the last to the first.

        '''
        # if self.cost_function == "cross_entropy" and self.layers[self.num_layers-1].activation_function == "softmax":
        targets = labels
        i = self.num_layers-1
        y = self.layers[i].activations
        deltab = np.multiply(y, np.multiply(1-y, targets-y))
        deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, deltab)
        new_weights = self.layers[i-1].weights_for_layer - self.learning_rate * deltaw
        new_bias = self.layers[i-1].bias_for_layer - self.learning_rate * deltab
        for i in range(i-1, 0, -1):
            y = self.layers[i].activations
            deltab = np.multiply(y, np.multiply(1-y, np.sum(np.multiply(new_bias, self.layers[i].bias_for_layer)).T))
            deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(y, np.multiply(1-y, np.sum(np.multiply(new_weights, self.layers[i].weights_for_layer),axis=1).T)))
            self.layers[i].weights_for_layer = new_weights
            self.layers[i].bias_for_layer = new_bias
            new_weights = self.layers[i-1].weights_for_layer - self.learning_rate * deltaw
            new_bias = self.layers[i-1].bias_for_layer - self.learning_rate * deltab
        self.layers[0].weights_for_layer = new_weights
        self.layers[0].bias_for_layer = new_bias


    def predict(self, filename, input):
        '''
        It takes 2 inputs:
        1. filename: The file from which trained model is to be loaded.
        2. input: The input for which we want the prediction.
        It does a forward pass, and then converts the output into one-hot encoding i.e. the maximum element of array is 1 and all others are 0.

        '''
        dill.load_session(filename)
        self.batch_size = 1
        self.forward_pass(input)
        a = self.layers[self.num_layers-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        return a

    def check_accuracy(self, filename, inputs, labels):
        '''
        It takes 3 inputs:
        1. filename: The file from which trained model is to be loaded.
        2. inputs: Input test data.
        3. labels: Target test data.

        This function does pretty much the same thing as predict. 
        But instead of returning the predicted output, it commpares the predictions with the labels, 
        and then calculates accuracy as correct*100/total.
'''

        dill.load_session(filename)
        self.batch_size = len(inputs)
        self.forward_pass(inputs)
        a = self.layers[self.num_layers-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        total=0
        correct=0
        for i in range(len(a)):
            total += 1
            if np.equal(a[i], labels[i]).all():
                correct += 1
        print("Accuracy: ", correct*100/total)



    def load_model(self, filename):
        dill.load_session(filename)


class layer:
    '''
     It takes 3 arguments:
     1. num_nodes_in_layer: Number of nodes in that layer.
     2. num_nodes_in_next_layer: Number of nodes in the next layer.
     3. activation_function: Activation function for that layer.

    This function is called from the constructor of neural_network class. 
    It initializes one layer at a time. The weights of the last layer are set to None.
'''
    def __init__(self, num_nodes_in_layer, num_nodes_in_next_layer, activation_function):
        self.num_nodes_in_layer = num_nodes_in_layer
        self.activation_function = activation_function
        self.activations = np.zeros([num_nodes_in_layer,1])
        if num_nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(0, 0.001, size=(num_nodes_in_layer, num_nodes_in_next_layer))
            self.bias_for_layer = np.random.normal(0, 0.001, size=(1, num_nodes_in_next_layer))
        else:
            self.weights_for_layer = None
            self.bias_for_layer = None