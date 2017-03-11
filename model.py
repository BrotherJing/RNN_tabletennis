import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.mlab as mlab
from dataloader import *
from util_MDN import *

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
		self.mixtures = config['mixtures']
		max_grad_norm = config['max_grad_norm']
		learning_rate = config['learning_rate']
		keep_prob = config['keep_prob']

		self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.coords, self.seq_len], name='Input_data')
		self.y_ = tf.placeholder(tf.float32, shape=[self.batch_size, self.coords, self.seq_len], name='Ground_truth')
		
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
		
		with tf.name_scope("MDN"):
			self.mixture_params = 8
			self.output_units = self.mixture_params * self.mixtures
			output = tf.reshape(tf.concat(outputs[:-1], 0), [-1, hidden_size])#[seqlen-1*batch_size, hidden_size]
			softmax_w = tf.get_variable("softmax_w", [hidden_size, self.output_units], dtype=tf.float32)
			softmax_b = tf.get_variable("softmax_b", [self.output_units], dtype=tf.float32, initializer=tf.constant_initializer(0.5))
			logits = tf.matmul(output, softmax_w) + softmax_b#[seqlen-1*batch_size, output_units]

			self._softmax_w = softmax_w

			h_xyz = tf.reshape(logits, (self.seq_len-1, self.batch_size, self.output_units))
			h_xyz = tf.transpose(h_xyz, [1,2,0])#[batch_size, output_units, seqlen-1]

			seq_delta = self.x[:,:3,1:] - self.x[:,:3,:-1]#ground truth [batch_size, 3, seqlen-1]
			delta1, delta2, delta3 = tf.split(seq_delta, 3, 1)#delta for x y z, each [batch_size, 1, seqlen-1]

			mu1, mu2, mu3, s1, s2, s3, rho, theta = tf.split(h_xyz, self.mixture_params, 1)#each is [batch_size, mixtures, seqlen-1]

			max_theta = tf.reduce_max(theta, 1, keep_dims=True)#max over all mixtures
			theta = tf.subtract(theta, max_theta)
			theta = tf.exp(theta)
			theta_norm = tf.reciprocal(tf.reduce_sum(theta, 1, keep_dims=True))
			theta = tf.multiply(theta_norm, theta)

			s1 = tf.exp(s1)
			s2 = tf.exp(s2)
			s3 = tf.exp(s3)#explode?
			rho = tf.tanh(rho)

			p_xy = tf_2d_normal(delta1, delta2, mu1, mu2, s1, s2, rho)
			p_z = tf_1d_normal(delta3, mu3, s3)
			p = tf.multiply(p_xy, p_z)#[batch_size, mixtures, seqlen-1] should be all [0,1]

			self._p_sum = p_sum = tf.reduce_sum(tf.multiply(p, theta), 1)#sum along the mixture dimension
			loss = -tf.log(tf.maximum(p_sum, 1e-20))

		self._cost = cost = tf.reduce_mean(loss)
		self._final_state = state
		self._outputs = [mu1, mu2, mu3, s1, s2, s3, rho, theta]

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
		self._lr_update = tf.assign(self._lr, self._new_lr)#this is an op

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr:lr_value})

	def sample_theta(self, thetas):
		stop = np.random.rand()#random number to stop, [0,1]
		num_thetas = len(thetas)#should be mixtures
		cum = 0.0
		for i in range(num_thetas):
			cum+=thetas[i]
			if cum>stop:
				return i
		return 0

	def sample(self, session, seq, sl_pre = 4, bias=1.0):
		assert seq.shape[1] == self.seq_len and seq.shape[0] == self.coords, 'Feed a sequence in [crd,sl]'
		assert sl_pre > 1, 'Please provide two predefined coordinates' 

		state = session.run(self._initial_state)

		seq_feed = np.zeros((self.batch_size, self.coords, self.seq_len))
		seq_feed[0,:,:] = seq[:,:]
		offset_draw = np.zeros((3))
		for sl_draw in range(sl_pre,self.seq_len-1):
			feed_dict = {self.x:seq_feed}
			for i, (c, h) in enumerate(self._initial_state):
				feed_dict[c] = state[i].c
				feed_dict[h] = state[i].h
			result = session.run([self._outputs], feed_dict=feed_dict)[0]

			idx_theta = self.sample_theta(result[7][0,:,sl_draw])
			
			mean = np.zeros((3))
			mean[0] = result[0][0,idx_theta,sl_draw]
			mean[1] = result[1][0,idx_theta,sl_draw]
			mean[2] = result[2][0,idx_theta,sl_draw]
			cov = np.zeros((3,3))
			s1 = np.exp(-1*bias)*result[3][0,idx_theta,sl_draw]
			s2 = np.exp(-1*bias)*result[4][0,idx_theta,sl_draw]
			s3 = np.exp(-1*bias)*result[5][0,idx_theta,sl_draw]
			s12 = result[6][0,idx_theta,sl_draw]*s1*s2
			cov[0,0] = np.square(s1)
			cov[1,1] = np.square(s2)
			cov[2,2] = np.square(s3)
			cov[0,1] = s12
			cov[1,0] = s12
			#cov[1,2] = s12
			#cov[2,1] = s12
			#print cov
			#cov += np.identity(3)*1e-6
			rv = multivariate_normal(mean, cov)
			draw = rv.rvs()
			seq_feed[0,:,sl_draw+1] = seq_feed[0,:,sl_draw] + draw
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
	def outputs(self):
		return self._outputs

	@property
	def p_sum(self):
		return self._p_sum

	@property
	def softmax_w(self):
		return self._softmax_w

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