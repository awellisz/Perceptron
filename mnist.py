import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

###################
#### FUNCTIONS ####
###################

def init_params(n_input_neurons, n_hidden_neurons, n_output_neurons):
    """
    init_params: Initialize weights and biases to random values in [-1,1]
    n_input/hidden/output_neurons: size of each layer of the neural network.
    """
    np.random.seed(12345)
    W1 = np.random.rand(n_hidden_neurons, n_input_neurons) - 0.5 
    b1 = np.random.rand(n_hidden_neurons, 1) - 0.5 
    W2 = np.random.rand(n_output_neurons, n_hidden_neurons) - 0.5
    b2 = np.random.rand(n_output_neurons, 1) - 0.5
    return W1, b1, W2, b2

def relu(Z):
    """
    relu: Rectified linear unit, the activation function used for the hidden layer
    """
    return np.maximum(Z, 0)

def softmax(Z):
    """
    softmax: The activation function used for the output layer. Converts a
             vector of numbers to a vector of probabilities (i.e. all elements
             end up in [0,1] and sum to 1).
    Z: Input array
    """
    return np.exp(Z) / sum(np.exp(Z))
    
def feedforward(W1, b1, W2, b2, X):
    """
    feedforward: Propagate training data through the model from input to 
                 predicted output to compute final layer values
    W1, W2: Weight matrices (input -> hidden, hidden -> output, respectively)
    b1, b2: Bias vectors
    X: Input data (feature matrix)
    """
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def relu_deriv(Z):
    """
    relu_deriv: Returns the derivative of relu(z): 1 if z > 0 and 0 otherwise
    """
    return np.where(Z > 0, 1, 0)

def encode_one_hot(y):
    """
    encode_one_hot: encodes the label vector as a "one-hot" matrix,
                    where each element K of the vector becomes an array where 
                    the K-th element is 1 and the rest are zero. This makes the
                    training output values the right matrix shape to work with 
                    the output layer activation.
    y: the input label vector. In this case, all labels are in [0,9], so each
        row of the one-hot array has 10 elements.
    """
    return (np.eye(np.max(y) + 1)[y]).T

def backpropagate(Z1, A1, Z2, A2, W1, b1, W2, b2, X, y, alpha):
    """
    backpropagation: Performs a backward pass to adjust NN weights according to
                     the backpropagation algorithm.
    Z1, A1: Hidden layer of NN
    Z2, A2: Output layer of NN
    W1, W2: Weights (input -> hidden, hidden -> output, respectively)
    b1, b2: Biases
    X: feature matrix
    y: label vector (i.e. actual output of training data)
    alpha: learning rate
    """
    m = X.shape[1]
    
    delta2 = A2 - encode_one_hot(y)
    dW2 = (1/m) * np.dot(delta2, A1.T)
    db2 = (1/m) * np.sum(delta2)
    
    delta1 = np.dot(W2.T, delta2) * relu_deriv(Z1)
    dW1 = (1/m) * np.dot(delta1, X.T)
    db1 = (1/m) * np.sum(delta1)
    
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2 
    
    return W1, b1, W2, b2

def get_predictions(A):
    """
    get_predictions: Finds the predicted label given the raw value of output layer.
    """
    return np.argmax(A, 0)

def get_accuracy(predictions, y):
    """
    get_accuracy: Calculates the accuracy of the neural network by comparing
                  actual training labels to predicted labels.
    predictions: predicted labels
    y: training data labels
    """
    return np.sum(predictions == y) / y.size

def print_progress_bar(curr, total, accuracy, bar_size = 30):
    """
    progress_bar: Prints a progress bar as the NN trains
    (Code adapted from StackOverflow)
    """
    pct = curr/total
    sys.stdout.write('\r')
    bar = '█' * int(bar_size * pct)
    bar = bar + '▒' * int(bar_size * (1-pct))
    accuracy_string = "{:.3f}".format(accuracy)
    itercounter = f'{curr}/{total}'
    sys.stdout.write(f"Accuracy: {accuracy_string.ljust(10)} | [{bar:{bar_size}s}] {int(100 * pct)}% ({itercounter})")
    sys.stdout.flush()

def train_network(X, y, alpha, num_epochs):
    """
    train_network: Brings everything together. Feeds forward and backpropagates
                   to iteratively find the optimal weights.
    X: training feature matrix
    y: training labels
    alpha: learning rate
    num_epochs: number of training iterations
    """

    print(f"\nTraining neural network with learning rate {alpha} over {num_epochs} training epochs...\n")

    W1, b1, W2, b2 = init_params(784, 80, 10)
    for i in range(num_epochs):
        Z1, A1, Z2, A2 = feedforward(W1, b1, W2, b2, X)
        W1, b1, W2, b2 = backpropagate(Z1, A1, Z2, A2, W1, b1, W2, b2, X, y, alpha)
        predictions = get_predictions(A2)
        print_progress_bar(i+1, num_epochs, get_accuracy(predictions, y_train))
    
    print("Training complete.\n")

    return W1, b1, W2, b2 

def test_network_on_img(index, W1, b1, W2, b2, X, y):
    """
    test_network_on_img: Uses the weights of the trained network and applies them
                         to a new test sample. Displays the image of the digit and
                         the predicted label + actual label.
    W1, b1, W2, b2: Calculated weights and biases of the trained NN.
    X, y: test dataset
    index: index of the test dataset to test and display.
    """
    if (index > X.shape[1]):
        print("Invalid index entered.")
        return

    img = X[:, index, None]

    _, _, _, A2 = feedforward(W1, b1, W2, b2, img)
    prediction = get_predictions(A2)
    label = y[index]

    img = img.reshape((28,28)) * 255 
    plt.gray() 
    plt.imshow(img, interpolation='nearest')
    plt.title(f"NN Prediction: {prediction}\nActual label: {label}")
    plt.show()

####################
### MAIN PROGRAM ###
####################

print("Reading CSVs...")
traindf = pd.read_csv("mnist_train.csv")
testdf = pd.read_csv("mnist_train.csv")
print("Reading complete.")

training_data = np.array(traindf).T
testing_data = np.array(testdf).T

y_train = training_data[0] # Labels
X_train = training_data[1:] # Features
# Normalize to be in [0, 1]
X_train = X_train / 255.

y_test = testing_data[0]
X_test = testing_data[1:]
X_test = X_test / 255.

W1, b1, W2, b2 = train_network(X_train, y_train, 0.2, 100)

while True:
    index = input(f"Enter an index from 0 to {X_test.shape[1]} to display the corresponding image and prediction. (-1 to quit).")
    index = int(index)
    if index == -1:
        break
    test_network_on_img(index, W1, b1, W2, b2, X_test, y_test)
