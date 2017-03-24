import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.mlab as mlab
from util_MDN import *

def sample_theta(thetas):
	stop = np.random.rand()#random number to stop, [0,1]
	num_thetas = len(thetas)#should be mixtures
	cum = 0.0
	for i in range(num_thetas):
		cum+=thetas[i]
		if cum>stop:
			return i
	return 0

def sample(session, placeholder_x, initial_state, outputs, config, seq, sl_pre = 4, bias=1.0):
	assert seq.shape[1] == config['seq_len'] and seq.shape[0] == config['coords'], 'Feed a sequence in [crd,sl]'
	assert sl_pre > 1, 'Please provide two predefined coordinates' 

	state = session.run(initial_state)

	seq_feed = np.zeros((config['batch_size'], config['coords'], config['seq_len']))
	seq_feed[0,:,:sl_pre+1] = seq[:,:sl_pre+1]
	offset_draw = np.zeros((3))
	for sl_draw in range(sl_pre,config['seq_len']-1):
		feed_dict = {placeholder_x:seq_feed}
		for i, (c, h) in enumerate(initial_state):
			feed_dict[c] = state[i][0]
			feed_dict[h] = state[i][1]
		result = session.run(outputs, feed_dict=feed_dict)

		idx_theta = sample_theta(result[7][0,:,sl_draw])
		
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

def sample_more(session, placeholder_x, initial_state, final_state, outputs, config, seq, predict_len=40, sl_pre = 4, bias=1.0):
	assert seq.shape[0] == config['coords'], 'Feed a sequence in [crd,sl]'
	assert sl_pre > 1, 'Please provide two predefined coordinates' 

	state = session.run(initial_state)

	seq_feed = np.zeros((config['batch_size'], config['coords'], (predict_len/config['seq_len']+1)*config['seq_len']))
	seq_feed[0,:,:sl_pre+1] = seq[:,:sl_pre+1]
	offset_draw = np.zeros((3))

	for i in range(sl_pre, predict_len):
		sl_draw = i % config['seq_len']#[0, seq_len-1]
		block = i/config['seq_len']#0,1,...
		feed_dict = {placeholder_x:seq_feed[:,:,block*config['seq_len']:(block+1)*config['seq_len']]}
		for j, (c, h) in enumerate(initial_state):
			feed_dict[c] = state[j][0]
			feed_dict[h] = state[j][1]
		if sl_draw == config['seq_len'] - 1:#last one
			result, state = session.run([outputs, final_state], feed_dict=feed_dict)
		else:
			result = session.run(outputs, feed_dict=feed_dict)

		idx_theta = sample_theta(result[7][0,:,sl_draw])
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
		cov[1,0] = s12
		cov[0,1] = s12
		rv = multivariate_normal(mean, cov)
		draw = rv.rvs()
		seq_feed[0,:,i+1] = seq_feed[0,:,i] + draw

	#for i in range(predict_len):
		#print seq_feed[0,0,i], seq_feed[0,1,i], seq_feed[0,2,i]

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.plot(seq[0,:], seq[1,:], seq[2,:],'r')
	ax.plot(seq_feed[0,0,:predict_len], seq_feed[0,1,:predict_len], seq_feed[0,2,:predict_len],'b')
	ax.set_xlabel('x coordinate')
	ax.set_ylabel('y coordinate')
	ax.set_zlabel('z coordinate')
	plt.show()