import numpy as np
import tensorflow as tf
import matplotlib as plt
from dataloader import *
from model import *

def run_epoch(session, model, X, y_, eval_op=None, n_epoch=0, summary=None, summary_writer=None, verbose=False):
	is_training = False
	costs = 0.0
	batch_size = config['batch_size']
	N, coords, _ = X.shape
	state = session.run(model.initial_state)
	fetches = {
		"cost": model.cost,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op
		is_training = True
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
		#state = vals['final_state']
		costs += cost
		if is_training:
			summary_str = session.run(summary, feed_dict=feed_dict)
			summary_writer.add_summary(summary_str, n_epoch*(N/batch_size+1)+step)
			summary_writer.flush()

	return costs

directory = 'data/'
filename = 'all_data.csv'#'seq_all.csv'

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
config['max_epoch'] = 60
config['max_max_epoch'] = 100

train_ratio = 0.8
plot_every = 100

plot = True

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

	#epochs = np.floor(config['batch_size']*max_iter/N)
	print "Train with %d epochs, %d iterations"%(config['max_max_epoch'], N*config['max_max_epoch']/config['batch_size'])

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config['init_scale'],
			config['init_scale'])

		with tf.name_scope("Train"):
			with tf.variable_scope("Model", reuse=None, initializer=initializer):
				model = Model(True, config)
			tf.summary.scalar("square error loss", model.cost)
			tf.summary.histogram("prob", model.p_sum)
			tf.summary.histogram("w", model.softmax_w)

		with tf.name_scope("Test"):
			with tf.variable_scope("Model", reuse=True, initializer=initializer):
				mtest = Model(False, config)
		summary = tf.summary.merge_all()
		init = tf.global_variables_initializer()
		#sv = tf.train.Supervisor(logdir=directory) tf.global_variables_initializer()
		with tf.Session() as session:
			summary_writer = tf.summary.FileWriter(directory, session.graph)
			session.run(init)
			for i in range(config['max_max_epoch']):
				lr_decay = config['lr_decay']**max(i+1-config['max_epoch'], 0.0)
				model.assign_lr(session, config['learning_rate']*lr_decay)
				print "Epoch: %d Learning rate: %.3f"%(i+1, session.run(model.lr))
				
				train_perplexity = run_epoch(session, model, X_train, y_train, eval_op=model.train_op, n_epoch=i+1, summary=summary, summary_writer=summary_writer, verbose=True)
				print "Epoch: %d train perplexity: %.3f"%(i+1, train_perplexity)

				test_perplexity = run_epoch(session, mtest, X_test, y_test)
				print "Epoch: %d test perplexity: %.3f"%(i+1, test_perplexity)

			model.sample(session, X_train[5], sl_pre=config['seq_len']/2);
			#print "saving model to %s"%(directory)
			#sv.saver.save(session, directory, global_step=sv.global_step)
if __name__ == '__main__':
	tf.app.run(main=main)