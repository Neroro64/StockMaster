import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import pickle

class Model:
    def __init__(self, nodes):
        """
        Arg:
            nodes - [in, n1, n2, ..., nj, out] an array of numbers of nodes in each layer. The first layer is an input layer, and the last is the output layer. 
        Returns:
            a fully-connected network with default weights and biases sampled from N(0, 1)
        """
        self.L = len(nodes)
        self.Nodes = nodes
        self.init_param()

    def init_param(self):
        """
        Initialize the weights and biases with samples from normal distributions
        W = [(n1xn2), (n2xn3), ..., (nj,nj+1)], where nj = number of nodes
        B = [(n1x1), (n2x1), ..., (njx1)]
        """
        self.W = [np.random.normal(0, 1, (i, j)) for i, j in zip(self.Nodes[:-1], self.Nodes[1:])]
        self.B = [np.random.normal(0, 1, (i,1)) for i in self.Nodes[1:]]

    def init_training_param(self, n, batch_size, epochs, lmbda, eta_min, eta_max, eta_size=0, cycles=0):
        """
        Initialize the hyper-parameters
        """
        self.N = n
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs if eta_size == 0 else int(cycles * 2 * eta_size)
        self.ETA_MIN = eta_min
        self.ETA_MAX = eta_max
        self.ETA_SIZE = eta_size
        self.BATCHES = int(n / batch_size)
        self.LAMBDA = lmbda
        self.ETA = eta_min

    def normalize(self, x):
        """
        Normalize the input data
        """
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True)
        x = (x - mean) / std
        return x
        
    def softmax(self, x):
        return np.exp(x-np.max(x, axis=0)) / np.sum(np.exp(x-np.max(x, axis=0)), axis=0)
    
    def cross_entropy(self, p, y):
        """
        Return the cross entroy cost of the prediction
        Note: a pseudo probability is assigned to where the predictions is zero, this is to avoid the invalid value error.
        """
        p[p==0] = 1e-7
        cost = 1/y.shape[1] * -np.sum(y*np.log(p))
        w_sum = [w**2 for w in self.W]
        s = 0
        for w in w_sum: 
            s += np.sum(w)
        cost += self.LAMBDA * s

        return cost

    def update_eta(self, t, l):
        """
        Update the learning rate for each update step
        """
        self.ETA = (t - 2*l*self.ETA_SIZE) / self.ETA_SIZE
        self.ETA = self.ETA * (self.ETA_MAX - self.ETA_MIN) + self.ETA_MIN

    def onehot(self, y):
        """
        converts y to one hot encoding
        """
        onehot = np.zeros((np.max(y)+1, len(y)))
        onehot[y, np.arange(len(y))] = 1
        return onehot
    
    def feedforward(self, activations):
        """
        s1 = w1 @ x + b1
        h1 = max(0, s1)
        ...
        sn = wn @ hn + bn
        return Softmax(sn)
        """

        a = activations[0]
        for i in range(self.L-1):
            s = self.W[i].T @ a + self.B[i]
            a = np.maximum(0, s)
            activations.append(a)

        p = self.softmax(a)
        return p

    def backPropagation(self, y, p, activations):
        """
        Back propagate the network and calculate the gradients
        """
        dw = [np.zeros(w.shape) for w in self.W]
        db = [np.zeros(b.shape) for b in self.B]

        g = -(y - p)
        for i in range(len(self.W)-1, -1, -1):
            dw[i] = g @ activations[i].T * 1/self.BATCH_SIZE + 2 * self.LAMBDA * self.W[i].T
            db[i] =  (np.sum(g, axis=1) * 1/self.BATCH_SIZE).reshape(self.B[i].shape)
            g = self.W[i] @ g
            g[np.where(activations[i]<=0)] = 0
        
        return (dw, db)
    
    def accuracy(self, p, y):
        """
        Compute the accuracy of the predictions
        """
        predictions = np.argmax(p, axis=0)
        y = np.argmax(y, axis=0)
        acc = predictions.T[predictions == y].shape[0] / p.shape[1]
        return acc
    
    def update_batch(self, x, y):
        """
        For each batch: 
            Pass the input into the network and compute the predictions.
            Back propagate through the network to compute the gradients using the stored activations
            Update the weights and biases using the gradients
        """
        activations = [x]
        p = self.feedforward(activations)
        dw, db = self.backPropagation(y, p, activations)

        for i in range(self.L-1):
            self.W[i] = self.W[i] - self.ETA * dw[i].T
            self.B[i] = self.B[i] - self.ETA * db[i]
        

    def SGD(self, training_data, test_data, log=False):
        """
        Stochastic gradient descend method
        Trains the network a given number of epochs or cycles
        Return:
            Training cost and validation cost
            Training accuracy and validation accuracy
        """
        x_t = training_data[0]
        x_v = test_data[0]
        y_t = training_data[1]
        y_v = test_data[1]

        training_cost = []
        validation_cost = []
        training_accuracy = []
        validation_accuracy = []

        t = 0
        k = (self.ETA_SIZE * 2) / 10
        while(t < self.EPOCHS):
            # Shuffles the order of samples 
            idx = np.random.permutation(self.N)

            for j in range(1, self.BATCHES):
                t += 1
                l = np.floor(t / (2 * self.ETA_SIZE))
                start = (j-1) * self.BATCH_SIZE
                end = j * self.BATCH_SIZE
                indices = idx[start:end]
                x_batch = x_t[:, indices]            
                y_batch = y_t[:, indices]            
                self.update_batch(x_batch, y_batch)
                self.update_eta(t, l)

                # Check cost and accuracy 10 times per cycle 
                if (t % k == 0):
                    p_t = self.feedforward([x_t])
                    p_v = self.feedforward([x_v])
                    training_cost.append(self.cross_entropy(p_t, training_data[1]))
                    validation_cost.append(self.cross_entropy(p_v, test_data[1]))
                    training_accuracy.append(self.accuracy(p_t, training_data[1]))
                    validation_accuracy.append(self.accuracy(p_v, test_data[1]))

            if (log):
                print("Epoch #{}--------------------------------------".format(t))
                print("Training Cost: {:.6f}".format(training_cost[-1]))
                print("Validation Cost: {:.6f}".format(validation_cost[-1]))
                print("Training Accuracy = {:.3f}".format(training_accuracy[-1]))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy[-1]))
                print("-"*50)

        return (training_cost, validation_cost, training_accuracy, validation_accuracy)

    def save(self, filename):
        """
        Save the model to the file 'filename`.
        """
        data = {"Nodes": self.Nodes,
                "W": [w.tolist() for w in self.W],
                "B": [b.tolist() for b in self.B]}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

    def load(self, filename):
        """
        Load the model
        """
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        self.Nodes = data["Nodes"]
        self.W = [np.array(w) for w in data["W"]]
        self.B = [np.array(b) for b in data["B"]]