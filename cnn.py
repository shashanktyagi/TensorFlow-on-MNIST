import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from pdb import set_trace as breakpoint

def convnet(X,weights,biases,p_dropout):
	images = tf.reshape(X,[-1,28,28,1])
	#conv layer
	layer_1 = tf.nn.bias_add(tf.nn.conv2d(images, weights['wc1'], strides=[1,1,1,1], padding='SAME'), biases['bc1'])
	layer_1 = tf.nn.max_pool(layer_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
	layer_1 = tf.nn.relu(layer_1)

	# conv layer
	layer_2 = tf.nn.bias_add(tf.nn.conv2d(layer_1, weights['wc2'], strides=[1,1,1,1], padding='SAME'), biases['bc2'])
	layer_2 = tf.nn.max_pool(layer_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
	layer_2 = tf.nn.relu(layer_2)

	# fc layer
	fc_input = tf.reshape(layer_2,[-1,7*7*64])
	layer_3 = tf.nn.bias_add(tf.matmul(fc_input,weights['wf1']), biases['bf1'])
	layer_3 = tf.nn.relu(layer_3)
	layer_3 = tf.nn.dropout(layer_3,p_dropout)
	
	#fc layer 
	layer_4 = tf.nn.bias_add(tf.matmul(layer_3,weights['wf2']), biases['bf2'])
	layer_4 = tf.nn.relu(layer_4)
	layer_4 = tf.nn.dropout(layer_4,p_dropout)

	# output layer
	output_layer = tf.nn.bias_add(tf.matmul(layer_3,weights['out']),biases['out'])

	return output_layer

def main():
	mnist = input_data.read_data_sets('./mnist', one_hot=True)

	weights = {
				'wc1': tf.Variable(0.001*tf.random_normal([5,5,1,32])),
				'wc2': tf.Variable(0.001*tf.random_normal([5,5,32,64])),
				'wf1': tf.Variable(0.001*tf.random_normal([7*7*64,1024])),
				'wf2': tf.Variable(0.001*tf.random_normal([1024,1024])),
				'out': tf.Variable(0.001*tf.random_normal([1024,10]))
	}

	biases = {
				'bc1': tf.Variable(0.001*tf.random_normal([32])),
				'bc2': tf.Variable(0.001*tf.random_normal([64])),
				'bf1': tf.Variable(0.001*tf.random_normal([1024])),
				'bf2': tf.Variable(0.001*tf.random_normal([1024])),
				'out': tf.Variable(0.001*tf.random_normal([10]))
	}
	
	learning_rate = 0.001
	beta1 = 0.9
	beta2 = 0.999
	num_epochs = 10
	batch_size = 100
	display_step = 1


	X = tf.placeholder(tf.float32,[None,784])
	Y = tf.placeholder(tf.float32,[None,10])
	p_dropout = tf.placeholder(tf.float32)
	
	pred = convnet(X,weights,biases, p_dropout)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
	
	optimizer = tf.train.AdamOptimizer(learning_rate,beta1,beta2).minimize(loss)

	total_loss = np.zeros(num_epochs)
	init = tf.global_variables_initializer()
	print 'starting iterations...'

	with tf.Session() as sess:
		sess.run(init)
		for epoch in xrange(num_epochs):
			curr_loss = 0
			num_batches = int(mnist.train.num_examples/batch_size)
			for i in xrange(num_batches):
				batch_X,batch_Y = mnist.train.next_batch(batch_size)
				#breakpoint()
				_,c = sess.run([optimizer,loss], feed_dict={X:batch_X,Y:batch_Y,p_dropout:0.75})
				curr_loss += c/batch_size
			total_loss[epoch] = curr_loss		
			if (epoch+1)%display_step == 0:
				print 'Epoch: %d\t cost: %.4f' % (epoch+1,curr_loss)

		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(Y,1)),tf.float32))		

		print 'Accuracy: %.4f' % (sess.run(accuracy, feed_dict={X:mnist.test.images,Y:mnist.test.labels,p_dropout:1}))
		plt.plot(total_loss,label='training loss')
		plt.legend()
		plt.show()

if __name__ == '__main__':
	main()








