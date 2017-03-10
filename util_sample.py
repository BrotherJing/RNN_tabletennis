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

def sample(session, placeholder_x, outputs, config, seq, sl_pre = 4, bias=1.0):
	assert seq.shape[1] == config['seq_len'] and seq.shape[0] == config['coords'], 'Feed a sequence in [crd,sl]'
	assert sl_pre > 1, 'Please provide two predefined coordinates' 

	seq_feed = np.zeros((config['batch_size'], config['coords'], config['seq_len']))
	seq_feed[0,:,:] = seq[:,:]
	offset_draw = np.zeros((3))
	for sl_draw in range(sl_pre,config['seq_len']-1):
		feed_dict = {placeholder_x:seq_feed}
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
		cov[1,2] = s12
		cov[2,1] = s12
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
