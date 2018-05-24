import numpy as np
import sys

class logistic_classifier(object):
    def __init__(self, coeff):
        # coeff is the regularization coefficient
        self.coeff = coeff
        # w will be our linear separator
        self.w = 0.0


    def compute_probabilities(self, Xtr):
        # function to compute mu_i (sigmoid(wTx)).
        # Complete this function. 
        pass
        
    def compute_loss(self, probabilities, Ytr):
        # function to compute training loss.
        # Keep in mind that log(x) tends to infinity if x is too close
        # to zero. Thus, whenever you think x could be close to
        # zero, add a small positive value (like 10^(-6)) to x so that
        # log doesn't return too big a value. 
        pass
        
    def compute_gradients(self, probabilities, Ytr, Xtr):
        # function to compute gradients with respect to w.
        # Complete it, keeping in mind the two gradient components -
        # the one for the logistic loss and the one for the regularizer.
        pass

    def update_weights(self, learning_rate, grads):
        # function to update weights. This needs to be filled in. 
        pass

    def sigmoid(self, inputs):
        # function to compute sigmoid of the inputs.
        # Keep in mind that if you exponentiate too large or too small
        # a value, the result might be out of bounds of computer
        # precision. Thus, put a lower and an upper cap on the
        # input to the exponential function. 
        pass
        
    def fit(self, Xtr, Ytr):
        '''
        This function trains the logistic regression model on the 
        given training data
        '''
        # num_iters: number of iterations that gradient descent
        # should run for.
        learning_rate = 0.00005
        
        self.num_iters = 10000

        # random initialization for w. 
        self.w = np.random.normal(0.0, 0.1, Xtr.shape[1])
        
        for iter in range(self.num_iters):
            probabilities = self.compute_probabilities(Xtr)
        
            train_loss = self.compute_loss(probabilities, Ytr)
            if iter % 1000 == 0:
                print "Train Loss = " + str(train_loss)
            
            grads = self.compute_gradients(probabilities, Ytr, Xtr)
            grads = grads / Xtr.shape[0]
            self.update_weights(learning_rate, grads)

    def predict(self, Xts):
        '''
        This function gives label predictions on the dataset fed to it. 
        '''
        linear_combinations = np.matmul(Xts, self.w)
        probabilities = self.sigmoid(linear_combinations)
        self.predictions = np.zeros(probabilities.shape)
        self.predictions = (probabilities > 0.5)
        return self.predictions


dataset = "spam"
# adjust the path according to where you have stored the dataset. 
path = "../" + dataset + "/"

Xtr = np.load(path + "Xtrain.npy")
Ytr = np.load(path + "Ytrain.npy")
Xts = np.load(path + "Xtest.npy")
Yts = np.load(path + "Ytest.npy")

print Xtr.shape, Xts.shape

model = logistic_classifier(coeff=0.0)
model.fit(Xtr, Ytr)
predictions = model.predict(Xts)

accuracy = 0.0
for i in range(len(predictions)):
    if predictions[i] == Yts[i]:
        accuracy += 1
accuracy /= len(predictions)
accuracy *= 100
test_accuracy = accuracy
print test_accuracy
