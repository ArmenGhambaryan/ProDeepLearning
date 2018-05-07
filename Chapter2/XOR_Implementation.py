# --------------------------------------
# XOR implementation in Tensorflow with hidden layers being sigmid to
# intorduce Non-Linearity
# --------------------------------------
import tensorflow as tf
# --------------------------------------
# Create placeholders for training input and output labels
# --------------------------------------
x_ = tf.placeholder(tf.float32, shape=[4,2], name = "x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name = "y-input")
# --------------------------------------
# Define the weights to the hidden and output layer respectively.
# --------------------------------------
w1 = tf.Variable(tf.random_uniform([2,2],-1,1), name = "Weights1")
w2 = tf.Variable(tf.random_uniform([2,1],-1,1), name = "Weights2")
# --------------------------------------
# Define the bias to the hidden and output layers respectively
# --------------------------------------
b1 = tf.Variable(tf.zeros([2]), name = "Bias1")
b2 = tf.Variable(tf.zeros([1]), name = "Bias2")
# --------------------------------------
# Define the final output through forward pass
# --------------------------------------
z2 = tf.sigmoid(tf.matmul(x_,w1)+b1)
pred = tf.sigmoid(tf.matmul(z2,w2)+b2)
# --------------------------------------
# Define the Cross-entropy/Log-loss Cost function based on the output label y and
# the predicted probability by the forward pass
# --------------------------------------
cost = tf.reduce_mean(((y_*tf.log(pred))+((1-y_)*tf.log(1.0-pred)))*-1)
learning_rate = 0.01
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# --------------------------------------
# Now that we have all that we need set up we will start the training
# --------------------------------------
XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [[0],[1],[1],[0]]
# --------------------------------------
# Initialize the variables
# --------------------------------------
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("./Downloads/XOR_logs",sess.graph)

sess.run(init)
for i in range(100000):
	sess.run(train_step, feed_dict={x_:XOR_X, y_:XOR_Y})

# --------------------------------------
print('Final Prediction', sess.run(pred, feed_dict={x_:XOR_X,y_:XOR_Y}))