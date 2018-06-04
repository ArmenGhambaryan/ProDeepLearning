import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import cv2
from slim.nets import vgg

batch_size = 32
width = 224
height = 224
learning_rate = 0.01
db_dir = os.path.join(os.getcwd(),'CatvsDog')
checkpoints_dir = os.path.join(os.getcwd(),'checkpoints')
slim = tf.contrib.slim
all_images = os.listdir(db_dir)
train_images, validation_images = train_test_split(all_images, train_size = 0.8, test_size = 0.2)
class_dicts = {1:'Cat',0:'Dog'}

MEAN_VALUE = np.array([103.939, 116.779, 123.68])

################################################
# Logic to read the images and also do mean correction
################################################

def image_preprocess(img_path,width,height):
	img = cv2.imread(img_path)
	img = imresize(img,(width,height))
	img = img - MEAN_VALUE
	return img

################################################
# Create generator for image batches so that only the batch
# is in memory
################################################

def data_gen_small(images, batch_size, width, height):
	while True:
		ix = np.random.choice(np.arange(len(images)), batch_size)
		imgs = []
		labels = []
		for i in ix:
			if images[i].split('.')[0] == 'cat':
				labels.append(1)
			else:
				if images[i].split('.')[0] == 'dog':
					labels.append(0)
			array_img = image_preprocess(os.path.join(db_dir,images[i]),width,height)
			imgs.append(array_img)
		imgs = np.array(imgs)
		labels = np.array(labels)
		labels = np.reshape(labels,(batch_size,1))
		yield imgs, labels

################################################
## Defining the generators for training and validation batches
################################################
train_gen = data_gen_small(train_images,batch_size,width,height)
val_gen = data_gen_small(validation_images,batch_size,width,height)

with tf.Graph().as_default():
	x = tf.placeholder(tf.float32,[None,width,height,3])
	y = tf.placeholder(tf.float32,[None,1])

################################################
## Load the VGG16 model from slim extract the fully connected layer
## before the final output layer
################################################
	with slim.arg_scope(vgg.vgg_arg_scope()):
		logits, end_points = vgg.vgg_16(x,num_classes = 1000,
			is_training = False)
		fc_7 = end_points['vgg_16/fc7']
## Define the only set of weights that we will learn W1 and b1
################################################
	W1 = tf.Variable(tf.random_normal([4096,1],mean=0.0,stddev=0.02),name = 'W1')
	b = tf.Variable(tf.random_normal([1],mean = 0.0,stddev=0.02),name = 'b')

################################################
	## Reshape the fully connected layer fc_7 and define
	## the logits and probability
################################################
	fc_7 = tf.reshape(fc_7,[-1,W1.get_shape().as_list()[0]])
	logitx = tf.nn.bias_add(tf.matmul(fc_7,W1),b)
	probx = tf.nn.sigmoid(logitx)

################################################
	# Define Cost and Optimizer
	# Only we wish to learn the weights W1 and b and hence include them in var_list
################################################
	
	cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logitx, labels = y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list=[W1,b])

################################################
	# Loading the pre-trained weights for VGG16
################################################
	init_fn = slim.assign_from_checkpoint_fn(
		os.path.join(checkpoints_dir,'vgg_16.ckpt'),
		slim.get_model_variables('vgg_16'))

################################################
	# Running the optimization for only 50 batches of size 32
################################################
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		init_fn(sess)
		# One epoch
		for i in range(1):
			for j in range(50):
				batch_x, batch_y = next(train_gen)
				val_x, val_y = next(val_gen)
				sess.run(optimizer, feed_dict={x:batch_x,y:batch_y})
				cost_train = sess.run(cost, feed_dict={x:batch_x,y:batch_y})
				cost_val = sess.run(cost,feed_dict={x:val_x,y:val_y})
				prob_out = sess.run(probx, feed_dict={x:val_x,y:val_y})
				print('Training Cost:',cost_train,'Validation Cost:',cost_val)
		out_val = (prob_out>0.5)*1
		print('Accuracy:',np.sum(out_val==val_y)*100/float(len(val_y)))
		plt.imshow(val_x[1] + MEAN_VALUE)
		print('Actual Class:',class_dict[val_y[1][0]])
		print('Predicted Class:',class_dict[out_val[1][0]])
		plt.imshow(val_x[2] + MEAN_VALUE)
		print('Actual Class:',class_dict[val_y[2][0]])
		print('Predicted Class:',class_dict[out_val[2][0]])
		plt.show()










