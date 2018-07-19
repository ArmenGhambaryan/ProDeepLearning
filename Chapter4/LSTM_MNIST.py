# Import the Required Libraries
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Batch Learning Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 50
num_train = mnist.train.num_examples
num_batches = (num_train // batch_size) + 1
epochs = 2

# RNN LSTM Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)

# Define the forward pass for the RNN

def RNN(x, weights, biases):
	# Unstack to get a list of n_steps tensors of shape (batch_size, n_input)
	x = tf.unstack(x, n_steps,1)

	# Define a lstm cell
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

	# Get lstm cell
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

# tf Graph input
x = tf.placeholder("float",[None, n_steps, n_input])
y = tf.placeholder("float",[None, n_classes])

# Define weights
weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
	'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
# Initilizing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		for step in range(num_batches):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			batch_x = batch_x.reshape((batch_size, n_steps, n_input))
		# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x:batch_x, y: batch_y})
			if (step + 1) % display_step == 0:
				# Calculate batch accuracy
				acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
				# Calculate batch loss
				loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
				print("Epoch: {:.0f}".format(epoch), "step: {:.0f}".format(step+1),
					"Minibatch Loss= {:.6f}".format(loss), "Training Accuracy= {:.5f}".format(acc))
	print('Optimization finished')

	# Calculate accuracy
	test_len = 500
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy: ", sess.run(accuracy, feed_dict={x:test_data, y: test_label}))
