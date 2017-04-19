import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./mnist', one_hot=True)

n_layer_1 = 256
n_layer_2 = 256
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
trainings_epochs = 20
batch_size = 100
display_step = 2


def multilayer_perceptron(X,Y,weights,biases):
	layer_1 = tf.add(tf.matmul(X,weights['w1']),biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	layer_2 = tf.add(tf.matmul(layer_1,weights['w2']),biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	output_layer = tf.add(tf.matmul(layer_1,weights['out']),biases['out'])
	return output_layer

weights = {
			'w1':tf.Variable(0.001*tf.random_normal([784,n_layer_1])),
			'w2':tf.Variable(0.001*tf.random_normal([n_layer_1,n_layer_2])),
			'out':tf.Variable(0.001*tf.random_normal([n_layer_2,10]))
}
biases = {
			'b1':tf.Variable(0.001*tf.random_normal([n_layer_1])),
			'b2':tf.Variable(0.001*tf.random_normal([n_layer_2])),
			'out':tf.Variable(0.001*tf.random_normal([10]))
}

X = tf.placeholder(tf.float32,[None,784])
Y = tf.placeholder(tf.float32,[None,10])
pred = multilayer_perceptron(X,Y,weights,biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2).minimize(loss)
init = tf.global_variables_initializer()

cost = np.zeros(trainings_epochs)
with tf.Session() as sess:
	sess.run(init)
	for epoch in xrange(trainings_epochs):
		num_batches = int(mnist.train.num_examples/batch_size)
		curr_cost = 0
		for i in xrange(num_batches):
			batch_X,batch_Y = mnist.train.next_batch(batch_size)
			_,c = sess.run([optimizer,loss], feed_dict={X:batch_X,Y:batch_Y})
			curr_cost += c/batch_size
		cost[epoch] = curr_cost
		if (epoch+1) % display_step == 0:
			print 'Epoch: %0d\t cost: %.4f' % (epoch+1,curr_cost)

	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(Y,1)), tf.float64))
	print "Accuracy: ", accuracy.eval({X:mnist.test.images, Y:mnist.test.labels})
	plt.plot(range(trainings_epochs),cost,label='training loss')
	plt.legend()
	plt.show()


