import numpy as np
import tensorflow as tf
import matplotlib as plt
from dataloader import *
from model import *

directory = 'data/'
filename = 'all_data.csv'#'seq_all.csv'

def run_epoch(session, model, X, y_, n_epoch=0, verbose=False):
	costs = 0.0
	batch_size = config['batch_size']
	N, coords, _ = X.shape
	state = session.run(model.initial_state)
	fetches = {
		"cost": model.cost,
	}
	for step in range(N/batch_size+1):
		batch_idx = np.random.choice(N, batch_size, replace=False)
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h
		feed_dict[model.x] = X[batch_idx]
		feed_dict[model.y_] = y_[batch_idx]
		vals = session.run(fetches, feed_dict)
		cost = vals['cost']
		costs += cost

	return costs

config = {}
config['num_layers'] = 2
config['hidden_size'] = 64
config['batch_size'] = 20#64
config['seq_len'] = 40#20
config['overlap_rate'] = 0.8#0.0
config['mixtures'] = 3
config['learning_rate'] = 0.005
config['lr_decay'] = 0.95
config['keep_prob'] = 1
config['max_grad_norm'] = 0.5
config['init_scale'] = 0.01
config['max_epoch'] = 6
config['max_max_epoch'] = 10

train_ratio = 0.8

def main(_):
	dl = DataLoad(directory, filename)

	dl.load_data(config['seq_len'], config['overlap_rate'], verbose = True)

	dl.split_train_test(train_ratio)

	data = dl.data
	X_train = np.transpose(data['X_train'], [0,2,1])
	y_train = np.transpose(data['y_train'], [0,2,1])
	X_test = np.transpose(data['X_test'], [0,2,1])
	y_test = np.transpose(data['y_test'], [0,2,1])

	N, coords, _ = X_train.shape
	N_test = X_test.shape[0]

	config['coords'] = coords

	print "Train with %d epochs, %d iterations"%(config['max_max_epoch'], N*config['max_max_epoch']/config['batch_size'])

	with tf.Graph().as_default():

		tf.reset_default_graph()

		saver = tf.train.import_meta_graph(directory+'my-model.meta')

		mtest = Model(False, config)

		init = tf.global_variables_initializer()

		with tf.Session() as session:
			session.run(init)
			saver.restore(session, directory+'my-model')

			for i in range(config['max_max_epoch']):
				
				test_perplexity = run_epoch(session, mtest, X_test, y_test)
				print "Epoch: %d test perplexity: %.3f"%(i+1, test_perplexity)

			model.sample(session, X_train[5], sl_pre=config['seq_len']/2);

if __name__ == '__main__':
	tf.app.run(main=main)