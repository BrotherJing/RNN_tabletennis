import numpy as np
import tensorflow as tf
import matplotlib as plt
from dataloader import *
from model import *
from util_sample import *
import os.path

directory = 'data/'
#filename = 'all_data.csv'
#filename = 'seq_all.csv'
filename = 'coords.csv'

config = {}
if filename=='seq_all.csv':
	config['seq_len'] = 20
	config['batch_size'] = 64
	config['overlap_rate'] = 0.0
	config['lr_decay'] = 0.9
	config['max_epoch'] = 10
	config['max_max_epoch'] = 20
elif filename =='all_data.csv':
	config['seq_len'] = 60
	config['batch_size'] = 64
	config['overlap_rate'] = 0.8
	config['lr_decay'] = 0.98
	config['max_epoch'] = 10
	config['max_max_epoch'] = 20
else:
	config['seq_len'] = 60
	config['batch_size'] = 64
	config['overlap_rate'] = 0.5
	config['lr_decay'] = 0.9
	config['max_epoch'] = 20
	config['max_max_epoch'] = 20
config['learning_rate'] = 0.005
config['num_layers'] = 2
config['hidden_size'] = 64
config['mixtures'] = 3
config['keep_prob'] = 0.9
config['max_grad_norm'] = 0.5
config['init_scale'] = 0.01

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

	g = tf.Graph()
	with g.as_default():

		output_graph_def = tf.GraphDef()
		output_graph_path = './data/export-graph.pb'

		with open(output_graph_path, 'rb') as f:
			output_graph_def.ParseFromString(f.read())
			_ = tf.import_graph_def(output_graph_def, name='')

		with tf.Session() as session:

			placeholder_x = session.graph.get_tensor_by_name("Test/Model/Input_data:0")
			initial_state_c_0 = session.graph.get_tensor_by_name("Test/Model/zeros:0")
			initial_state_h_0 = session.graph.get_tensor_by_name("Test/Model/zeros_1:0")
			initial_state_c_1 = session.graph.get_tensor_by_name("Test/Model/zeros_2:0")
			initial_state_h_1 = session.graph.get_tensor_by_name("Test/Model/zeros_3:0")
			initial_state = []
			initial_state.append((initial_state_c_0, initial_state_h_0))
			initial_state.append((initial_state_c_1, initial_state_h_1))
			final_state_c_0 = session.graph.get_tensor_by_name("Test/Model/RNN/multi_rnn_cell_59/cell_0/lstm_cell/add_3:0")
			final_state_h_0 = session.graph.get_tensor_by_name("Test/Model/RNN/multi_rnn_cell_59/cell_0/lstm_cell/mul_5:0")
			final_state_c_1 = session.graph.get_tensor_by_name("Test/Model/RNN/multi_rnn_cell_59/cell_1/lstm_cell/add_3:0")
			final_state_h_1 = session.graph.get_tensor_by_name("Test/Model/RNN/multi_rnn_cell_59/cell_1/lstm_cell/mul_5:0")
			final_state = []
			final_state.append((final_state_c_0, final_state_h_0))
			final_state.append((final_state_c_1, final_state_h_1))
			placeholder_x = session.graph.get_tensor_by_name("Test/Model/Input_data:0")
			test_outputs = []
			test_outputs.append(session.graph.get_tensor_by_name("Test/Model/MDN/split_1:0"))
			test_outputs.append(session.graph.get_tensor_by_name("Test/Model/MDN/split_1:1"))
			test_outputs.append(session.graph.get_tensor_by_name("Test/Model/MDN/split_1:2"))
			test_outputs.append(session.graph.get_tensor_by_name("Test/Model/MDN/Exp_1:0"))
			test_outputs.append(session.graph.get_tensor_by_name("Test/Model/MDN/Exp_2:0"))
			test_outputs.append(session.graph.get_tensor_by_name("Test/Model/MDN/Exp_3:0"))
			test_outputs.append(session.graph.get_tensor_by_name("Test/Model/MDN/Tanh:0"))
			test_outputs.append(session.graph.get_tensor_by_name("Test/Model/MDN/Mul:0"))

			for i in range(config['max_epoch']):
				sample_more(session, placeholder_x, initial_state, final_state, test_outputs, config, X_test[np.random.choice(N_test)], predict_len=120, sl_pre=config['seq_len']/2)

if __name__ == '__main__':
	tf.app.run(main=main)