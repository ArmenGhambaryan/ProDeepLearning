# --------------------------------------
# Import the required libraries
# --------------------------------------

import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.examples.tutorials.mnist import input_data

# --------------------------------------
# Function to read the MNIST dataset along with the labels
# --------------------------------------

def read_infile():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
	train_X, train_Y, test_X, test_Y = mnist.train.images, mnist.train.labels, \
	mnist.test.images, mnist.test.labels
	return train_X, train_Y, test_X, test_Y

# --------------------------------------
# Define the weights and biases for the neural network
# --------------------------------------

def weights_biases_placeholder(n_dim, n_classes):
	X = tf.placeholder(tf.float32,[None, n_dim])
	Y = tf.placeholder(tf.float32,[None, n_classes])
	w = tf.Variable(tf.random_normal([n_dim,n_classes],stddev=0.01),name='weights')
	b = tf.Variable(tf.random_normal([n_classes]),name='weights')
	return X,Y,w,b

# --------------------------------------
# Define the forward pass
# --------------------------------------

def forward_pass(w,b,X):
	out = tf.matmul(X,w) + b
	return out

# --------------------------------------
# Define the cost function for the SoftMax unit
# --------------------------------------

def multiclass_cost(out,Y):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels = Y))
	return cost

# --------------------------------------
# Define the initialization op
# --------------------------------------

def init():
	return tf.global_variables_initializer()

# --------------------------------------
# Define the training op
# --------------------------------------

def train_op(learning_rate,cost):
	op_train = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	return op_train

train_X, train_Y, test_X, test_Y = read_infile()
X,Y,w,b = weights_biases_placeholder(train_X.shape[1], train_Y.shape[1])
out = forward_pass(w,b,X)
cost = multiclass_cost(out, Y)
learning_rate,epochs, batch_size = 0.01,1000,1000
num_batches = train_X.shape[0]//batch_size
op_train = train_op(learning_rate, cost)
init = init()
epoch_cost_trace = []
epoch_accuracy_trace = []

# --------------------------------------
# Activate the tensorFlow session and execute the stochastic gradient descent
# --------------------------------------

with tf.Session() as sess:
	sess.run(init)
	for i in range(epochs):
		epoch_cost, epoch_accuracy = 0,0
		for j in range(num_batches):
			sess.run(op_train, feed_dict={X:train_X[j*batch_size:(j+1)*batch_size],
				Y:train_Y[j*batch_size:(j+1)*batch_size]})
			actual_batch_size = train_X[j*batch_size:(j+1)*batch_size].shape[0]
			epoch_cost += actual_batch_size*sess.run(cost,feed_dict=\
				{X:train_X[j*batch_size:(j+1)*batch_size],Y:train_Y[j*batch_size:(j+1)*batch_size]})
			epoch_cost = epoch_cost/float(train_X.shape[0])
			epoch_accuracy = np.mean(np.argmax(sess.run(out,feed_dict={X:train_X,Y:train_Y}),
				axis=1)==np.argmax(train_Y,axis=1))
			epoch_cost_trace.append(epoch_cost)
			epoch_accuracy_trace.append(epoch_accuracy)

			if (((i+1) >= 100) and ((i+1) % 100 == 0)):
				print('Epoch:',(i+1),'Average loss:',epoch_cost,'accuracy:',epoch_accuracy)
	print('Final epoch training results:','Average loss:',epoch_cost,'accuracy:',epoch_accuracy)
	loss_test = sess.run(cost, feed_dict={X:test_X,Y:test_Y})
	test_pred = np.argmax(sess.run(out,feed_dict={X:test_X,Y:test_Y}),axis=1)
	accuracy_test = np.mean(test_pred == np.argmax(test_Y,axis=1))
	print('Results on test dataset:','Average loss:',loss_test,'accuracy:',accuracy_test)

import matplotlib.pyplot as plt
f,a = plt.subplots(1,10, figsize=(10,2))
print('Actual digits ',np.argmax(test_Y[0:10],axis=1))
print('Predicted digits:',test_pred[0:10])
print('Actual images of the digits follow:')
for i in range(10):
	a[i].imshow(np.reshape(test_X[i],(28,28)))
