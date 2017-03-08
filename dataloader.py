import numpy as np
import pandas as pd

class DataLoad():
	def __init__(self, dir, filename):
		"""
		input:
		- dir: directory where the data is stored
		- filename: name of the data file
		"""
		self.data_path = dir+filename
		self.X = []
		self.labels = []
		self.data = {}#will be divided into train/val/test
		self.N = 0
		self.iter_train = 0
		self.epochs = 0

	def load_data(self, seq_len = 20, overlap_rate = 0.2, verbose = False):
		if self.X:
			print "You already have the data"
			return
		df = pd.read_csv(self.data_path)
		if verbose:
			print "the shape of the data is ", df.shape
			#test = df[df['id']=='10']
			#test.head(10)

		df_arr = df.as_matrix(['x','y','z','t'])#'rankc'
		df = None
		start_idx = 0
		N,D = df_arr.shape#N*4
		for i in range(1,N,1):
			if verbose and i%1000==0:
				print "load %5d of %5d"%(i,N)
			if int(df_arr[i,3])==1:#encounter a new sequence
				end_idx = i
				seq = df_arr[start_idx:end_idx,:]
				seq[:,2] = seq[:,2] - np.min(seq[:,2])#the default height
				seq = seq/1000
				while seq.shape[0]>=seq_len+1:
					self.X.append(seq[:seq_len,:3])
					self.labels.append(seq[1:seq_len+1,:3])
					seq = seq[int(seq_len*(1.0 - overlap_rate)):]
				start_idx = end_idx
		if verbose:
			print "load %d sequences of data"%len(self.X)
		self.X = np.stack(self.X, 0)
		self.labels = np.stack(self.labels, 0)

	def split_train_test(self, train_ratio):
		assert not isinstance(self.X, list), 'first load the data'
		N, seq_len, coords = self.X.shape
		assert seq_len > 1, 'sequence length should be greater than 1'
		assert train_ratio < 1.0, 'invalid train/test ratio'
		idx_cut = int(train_ratio*N)
		self.data['X_train'] = self.X[:idx_cut]
		self.data['y_train'] = self.labels[:idx_cut]
		self.data['X_test'] = self.X[idx_cut:]
		self.data['y_test'] = self.labels[idx_cut:]
		print "%d train samples and %d test samples"%(idx_cut, N - idx_cut)