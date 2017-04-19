import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist', one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 100


X = tf.placeholder(tf.float64,[None,784])
Y = tf.placeholder(tf.float64,[None,10])

W = tf.Variable(0.001*np.random.randn(784,10), name='weights')
b = tf.Variable(0.001*np.random.randn(10), name='bias')

pred = tf.nn.softmax(tf.matmul(X,W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(pred),reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in xrange(training_epochs):
		num_batches = int(mnist.train.num_examples/batch_size)
		loss = 0
		for batch in xrange(num_batches):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			_,c = sess.run([optimizer,cost], feed_dict={X:batch_x,Y:batch_y})
			loss += c/batch_size

		print 'Epoch: %d\t cost: %.4f' % (epoch+1,loss)


	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(Y,1)), tf.float64))
	print "Accuracy: ", accuracy.eval({X:mnist.test.images[:3000], Y:mnist.test.labels[:3000]})











