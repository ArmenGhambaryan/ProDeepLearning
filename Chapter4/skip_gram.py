import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 

#--------------------------------------------
# Function to one-hot encode the words
#--------------------------------------------

def one_hot(ind, vocab_size):
	rec = np.zeros(vocab_size)
	rec[ind] = 1
	return rec

#--------------------------------------------
# Function to create the training data from corpus
#--------------------------------------------

def create_training_data(corpus_raw, WINDOW_SIZE = 2):
	words_list = []
	for sent in corpus_raw.split('.'):
		for w in sent.split():
			if w != '.':
				words_list.append(w.split('.')[0])
	words_list = set(words_list)
	word2ind = {}
	ind2word = {}
	vocab_size = len(words_list)

	for i,w in enumerate(words_list):
		word2ind[w] = i
		ind2word[i] = w
	print(word2ind)
	sentences_list = corpus_raw.split('.')
	sentences = []

	for sent in sentences_list:
		sent_array = sent.split()
		sent_array = [s.split('.')[0] for s in sent_array]
		sentences.append(sent_array)
	data_recs = []
	for sent in sentences:
		for ind, w in enumerate(sent):
			for nb_w in sent[max(ind - WINDOW_SIZE,0):min(ind+WINDOW_SIZE,len(sent))+1]:
				if nb_w != w:
					data_recs.append([w,nb_w])
	x_train, y_train = [], []
	for rec in data_recs:
		x_train.append(one_hot(word2ind[rec[0]],vocab_size))
		y_train.append(one_hot(word2ind[rec[1]],vocab_size))

	return x_train, y_train, word2ind, ind2word, vocab_size

def read_data():
	with open('data.txt','r') as f:
		corpus_raw = f.read()
	return corpus_raw

corpus_raw = read_data().lower()
x_train, y_train, word2ind, ind2word, vocab_size = create_training_data(corpus_raw,2)

#--------------------------------------------
# Define Tensorflow ops and variable, invoke training
#--------------------------------------------
emb_dims = 128
learning_rate = 0.001
#--------------------------------------------
# Placeholders for Input output
x = tf.placeholder(tf.float32,[None, vocab_size])
y = tf.placeholder(tf.float32,[None,vocab_size])
#--------------------------------------------
# Define the embedding matrix weights and a bias
#--------------------------------------------
W = tf.Variable(tf.random_normal([vocab_size,emb_dims],mean = 0.0, stddev = 0.02,dtype = tf.float32))
b = tf.Variable(tf.random_normal([emb_dims],mean = 0.0, stddev = 0.02, dtype = tf.float32))
W_outer = tf.Variable(tf.random_normal([emb_dims,vocab_size],mean = 0.0, stddev = 0.02, dtype = tf.float32))
b_outer = tf.Variable(tf.random_normal([vocab_size],mean = 0.0, stddev = 0.02, dtype = tf.float32))

hidden = tf.add(tf.matmul(x,W),b)
logits = tf.add(tf.matmul(hidden, W_outer), b_outer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

epochs, batch_size = 100, 10
batch = len(x_train)//batch_size

# train for n_iter iterations
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for epoch in range(epochs):
		batch_index = 0
		for batch_num in range(batch):
			x_batch = x_train[batch_index:batch_index + batch_size]
			y_batch = y_train[batch_index:batch_index + batch_size]
			batch_index += batch_size
			sess.run(optimizer,feed_dict={x:x_batch, y:y_batch})
			print('epoch:', epoch,'loss:', sess.run(cost, feed_dict={x:x_batch,y:y_batch}))
	W_embed_trained = sess.run(W)
W_embedded = TSNE(n_components=2).fit_transform(W_embed_trained)*10
plt.figure(figsize=(10,10))
for i in range(len(W_embedded)):
	plt.text(W_embedded[i,0],W_embedded[i,1],ind2word[i])
plt.xlim(-150,150)
plt.ylim(-150,150)
plt.show()