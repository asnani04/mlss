Programming Assignment 1 - Linear Classification
------------------------------------------------

We continue with the SPAM dataset that we used while practising classification using k-Nearest Neighbors and Decision Trees in Lecture 2. This dataset comprises of emails, where each email is represented by a 57 dimensional feature vector. The training set consists of 3680 emails and the test set has the remaining 921 emails. Here, we would implement a linear classification model to try and classify emails into spam and non-spam.

You have to implement a logistic regression model for this purpose. It would be preferred if you use Python for the same, although other convenient languages like Octave can also be used. If you implement the model from scratch (with or without the skeleton code provided in this example), you will get more credit than if you were to use a library that already has logistic regression implemented (like sklearn). Few marks are awarded for accuracy, so don't worry if your model doesn't perform very well. Just make sure that it is correct (that is where the bulk of the marks are) and not very inefficient. 

The skeleton code is provided in log_reg_skeleton.py. Try to understand how it is supposed to work, and code up the parts that have been left for you to fill (these are some functions that belong to the linear_classifier class). Record your observations (training / test accuracy / training loss) for specific values of learning_rate and num_iters for both, regularized and unregularized linear classification. Also give reasons for any pattern that you observe among the observations.

Note: You are free to use the library Numpy for your code. 

Deliverables: Completed code and a small report (<= 1 page) highlighting the observations and conclusions. 

Deadline: Monday 28th May, 11:59 pm. 
