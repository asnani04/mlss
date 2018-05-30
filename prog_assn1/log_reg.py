import numpy as np
import sys

class logistic_classifier(object):
    def __init__(self, coeff):
        # coeff is the regularization coefficient
        self.coeff = coeff
        self.w = 0.0


    def compute_probabilities(self, Xtr):
        linear_combinations = np.matmul(Xtr, self.w)
        # print linear_combinations
        probabilities = self.sigmoid(linear_combinations)
        return probabilities

    def compute_loss(self, probabilities, Ytr):
        term1 = -np.multiply(Ytr, np.log(probabilities + 0.000001))
        term2 = -np.multiply((1.0 - Ytr), np.log(1.0 - probabilities +
                                                 0.000001))
        logistic_loss = np.sum(term1 + term2)
        reg_loss = 0.5 * self.coeff * (np.linalg.norm(self.w) ** 2)
        return logistic_loss + reg_loss
        
    def compute_gradients(self, probabilities, Ytr, Xtr):
        logistic_grads = np.matmul(np.transpose(Xtr), (probabilities -
                                                       Ytr))
        reg_grads = self.coeff * self.w
        return logistic_grads + reg_grads

    def update_weights(self, learning_rate, grads):
        self.w = self.w - learning_rate * grads
        return

    def sigmoid(self, inputs):
        inputs = np.minimum(inputs, np.ones_like(inputs)*5.0)
        inputs = np.maximum(inputs, -np.ones_like(inputs)*5.0)
        return 1.0 / (1.0 + np.exp(-inputs))
        
    def fit(self, Xtr, Ytr):
        '''
        This function trains the logistic regression model on the 
        given training data
        '''
        # num_iters: number of iterations that gradient descent
        # should run for.
        learning_rate = 0.00005
        
        self.num_iters = 50000
        self.w = np.random.normal(0.0, 0.1, Xtr.shape[1])
        # print self.w
        for iter in range(self.num_iters):
            probabilities = self.compute_probabilities(Xtr)
            # print probabilities
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

model = logistic_classifier(coeff=1.0)
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
