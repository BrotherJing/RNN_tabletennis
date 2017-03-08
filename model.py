import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.mlab as mlab
from dataloader import *
'''
flags = tf.flags
logging = tf.logging

flags.DEFINE_string('model','small','small, medium or large.')
flags.DEFINE_bool('use_fp16', False, 'train using 16-bit floats instead of 32-bit')

FLAGS = flags.FLAGS

def data_type():
	return tf.float16 if FLAGS.use_fp16 else tf.float32
'''
class Model():
	def __init__(self, is_training, config):
		num_layers = config['num_layers']
		hidden_size = config['hidden_size']
		self.batch_size = config['batch_size']
		self.seq_len = config['seq_len']
		self.coords = config['coords']
		max_grad_norm = config['max_grad_norm']
		learning_rate = config['learning_rate']
		keep_prob = config['keep_prob']

		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.coords, self.seq_len], name='Input_data')
		self.y_ = tf.placeholder(tf.float32, shape=[self.batch_size, self.coords, self.seq_len], name='Ground_truth')
		#self.keep_prob = tf.placeholder(tf.float32)
		
		def lstm_cell():
			return tf.contrib.rnn.LSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True, use_peepholes=True)
		attn_cell = lstm_cell
		if is_training and keep_prob<1:
			def attn_cell():
				return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=keep_prob)
		cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(num_layers)], state_is_tuple=True)

		self._initial_state = cell.zero_state(self.batch_size, tf.float32)

		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for timestep in range(self.seq_len):
				if timestep>0: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(self.x[:,:,timestep], state)
				outputs.append(cell_output)
		output = tf.reshape(tf.concat(outputs, 0), [-1, hidden_size])
		softmax_w = tf.get_variable("softmax_w", [hidden_size, self.coords], dtype=tf.float32)
		softmax_b = tf.get_variable("softmax_b", [self.coords], dtype=tf.float32)
		logits = tf.matmul(output, softmax_w) + softmax_b
		loss = tf.square(tf.subtract(logits, tf.reshape(tf.transpose(self.y_, [2,0,1]), [-1, self.coords])))
		self._cost = cost = tf.reduce_sum(loss)/self.batch_size
		self._final_state = state
		self._outputs = logits

		if not is_training:
			return

		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
			max_grad_norm)
		optimizer = tf.train.AdamOptimizer(self._lr)
		gradients = zip(grads, tvars)
		self._train_op = optimizer.apply_gradients(gradients, global_step=tf.contrib.framework.get_or_create_global_step())
		self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr:lr_value})

	def sample(self, session, seq, sl_pre = 4):
		assert seq.shape[1] == self.seq_len and seq.shape[0] == self.coords, 'Feed a sequence in [crd,sl]'
		assert sl_pre > 1, 'Please provide two predefined coordinates' 

		seq_feed = np.zeros((self.batch_size, self.coords, self.seq_len))
		seq_feed[0,:,:] = seq[:,:]
		offset_draw = np.zeros((3))
		for sl_draw in range(sl_pre,self.seq_len-1):
			feed_dict = {self.x:seq_feed}
			result = session.run([self._outputs], feed_dict=feed_dict)
			result = np.reshape(result[0], [self.batch_size, self.coords, self.seq_len])
			coordinates = result[0,:,sl_draw]
			seq_feed[0,:,sl_draw+1] = coordinates
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.plot(seq[0,:], seq[1,:], seq[2,:],'r')
		ax.plot(seq_feed[0,0,:], seq_feed[0,1,:], seq_feed[0,2,:],'b')
		ax.set_xlabel('x coordinate')
		ax.set_ylabel('y coordinate')
		ax.set_zlabel('z coordinate')
		plt.show()

	@property
	def cost(self):
		return self._cost

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

	@property
	def final_state(self):
		return self._final_state

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def x(self):
		return self.x

	@property
	def y_(self):
		return self.y_