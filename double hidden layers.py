import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# size of hidden layer
hidden_size1 = 20
hidden_size2 = 30

# learning rate
alpha = 0.1

iters = 600

# regularization strength
reg = 1e-6

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffling before splitting into test and training sets (to minimize risk of over fitting)

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test  = X_test / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255
_,m_train = X_train.shape

def init_params():
    W1 = np.random.normal(size=(hidden_size1, 784)) * np.sqrt(1/784)
    b1 = np.random.normal(size=(hidden_size1, 1)) * np.sqrt(1/10)
    W2 = np.random.normal(size=(hidden_size2, hidden_size1)) * np.sqrt(1/20)
    b2 = np.random.normal(size=(hidden_size2, 1)) * np.sqrt(1/10)
    W3 = np.random.normal(size=(10, hidden_size2)) * np.sqrt(1/20)
    b3 = np.random.normal(size=(10, 1)) * np.sqrt(1/10)

    # W1 = 1/hidden_size * np.random.randn(hidden_size, 784)
    # b1 = np.zeros((hidden_size, 1))
    # W2 = 1/hidden_size * np.random.randn(10, hidden_size)
    # b2 = np.zeros((10, 1))

    return W1, b1, W2, b2, W3, b3

def tanh(Z):
    return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

def tanh_deriv(Z):
    return 1 - tanh(Z) ** 2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z -= np.max(Z)
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def sigmoid(Z):
    try:
        return 1 / (1 + np.exp(-Z))
    except:
        return 0
    
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):  
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    Y_new = one_hot(Y)
    dZ3 = A3 - Y_new
    dW3 = 1/m * dZ3.dot(A2.T)
    db3 = 1/m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1)

    # penalty on large values inside W (L2 regularization)
    dW3 += reg * W3
    dW2 += reg * W2
    dW1 += reg * W1

    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X_train, Y_train, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):

        # taking mini-batches (yields slightly worse results)
        # random_indices = np.random.choice(X_train.shape[1], size=100, replace=False)
        # X = X_train[:, random_indices]
        # Y = Y_train[random_indices]
        # Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        # dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)    

        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_train)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X_train, Y_train)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if (10*i/iterations) % 1 == 0 or i==iterations-1:
            print("Iteration: ", i if i!=iterations-1 else i+1)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, Y_train))
    return W1, b1, W2, b2, W3, b3


W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, alpha, iters)


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


test_prediction(0, W1, b1, W2, b2, W3, b3)


test_predictions = make_predictions(X_test, W1, b1, W2, b2, W3, b3)
print("Accuracy on the test data:",get_accuracy(test_predictions, Y_test))
