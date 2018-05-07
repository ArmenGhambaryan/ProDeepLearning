# --------------------------------------
# Importing TensorFlow, Numpy, and the Boston Housing price dataset
# --------------------------------------

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston

# --------------------------------------
# Function to load the Boston data set
# --------------------------------------

def read_infile():
	data = load_boston()
	features = np.array(data.data)
	target = np.array(data.target)
	return features, target

# --------------------------------------
# Normalize the features by Z scaling; i.e., subtract from each feature value its mean and then
# devide by its standard deviation. Accelerates gradient descent.
# --------------------------------------

def feature_normalize(data):
	mu = np.mean(data, axis=0)
	std = np.std(data, axis = 0)
	return (data - mu)/std

# --------------------------------------
# Append the feature for the bias term
# --------------------------------------

def append_bias(features,target):
	n_samples = features.shape[0]
	n_features = features.shape[1]
	intercept_feature = np.ones((n_samples,1))
	X = np.concatenate((features, intercept_feature),axis=1)
	X = np.reshape(X,[n_samples,n_features + 1])
	Y = np.reshape(target,[n_samples,1])
	return X,Y

# --------------------------------------
# Execute the functions to read, normalize, and add append bias term to the data
# --------------------------------------

features, target = read_infile()
z_features = feature_normalize(features)
X_input, Y_input = append_bias(z_features, target)
num_features = X_input.shape[1]

# --------------------------------------
# Create TensorFlow ops for placeholders, weights, and weight initilization
# --------------------------------------

X = tf.placeholder(tf.float32,[None,num_features])
Y = tf.placeholder(tf.float32,[None,1])
w = tf.Variable(tf.random_normal((num_features,1)),name='weights')
init = tf.global_variables_initializer()

# --------------------------------------
# Define the different TensorFlow ops and input parameters for Cost and Optimization.
# --------------------------------------

learning_rate = 0.01
num_epochs = 1000
cost_trace = []
pred = tf.matmul(X,w)
error = pred - Y
cost = tf.reduce_mean(tf.square(error))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# --------------------------------------
# Execute the gradient-descent learning
# --------------------------------------

with tf.Session() as sess:
	sess.run(init)
	for i in range(num_epochs):
		sess.run(train_op, feed_dict={X:X_input,Y:Y_input})
		cost_trace.append(sess.run(cost,feed_dict={X:X_input,Y:Y_input}))
	error_ = sess.run(error,{X:X_input,Y:Y_input})
	pred_ = sess.run(pred,{X:X_input})

print('MSE in training:',cost_trace[-1])

# --------------------------------------
# Plot the reduction in cost over iterations or epochs
# --------------------------------------

import matplotlib.pyplot as plt
plt.plot(cost_trace)

# --------------------------------------
# Plot the Predicted House Prices vs the Actual House Prices
# --------------------------------------

fig, ax = plt.subplots()
plt.scatter(Y_input, pred_)
ax.set_xlabel('Actual House price')
ax.set_ylabel('Predicted House price')
plt.show()