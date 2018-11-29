'''
Simple ANN API

# Methods:
1. train_test_dataset()

'''

import os
import sys

import pandas as pd
import numpy as np

# from sklearn.model_selection import train_test_split

from keras import models
from keras import layers


class ML(object):

	# Dunders
	def __init__(self, df= None, name= "ML", labels= None, test_size= None, n_layers= None, n_nodes_per_layer= None, actvn_per_layer= None, lr_rate= None, optimizer= None, loss= None, metrics= None, epochs= None, batch_size= None): # df: dataframe to train on, label: label column/s in the dataset-> this is a list of column names
		self.name= name
		self.df= df
		self.labels= labels

		self.test_size= test_size
		self.n_layers= n_layers
		self.n_nodes_per_layer= n_nodes_per_layer
		self.actvn_per_layer= actvn_per_layer

		self.lr_rate= lr_rate
		self.optimizer= optimizer
		self.loss= loss
		self.metrics= metrics
		self.epochs= epochs
		self.batch_size= batch_size


	def __repr__(self): # Gives some information on the dataframe
		return("DataFrame with columns {} and shape {} to be trained for predicting labels {}, using {}.".format(self.df.columns, self.df.shape, self.labels, self.name))

	def __len__(self): # Gives the number of rows
		return self.df.shape[0]


	# Getters
	@property
	def df(self):
		return self._df
	
	@property
	def labels(self):
		return self._labels
	
	@property
	def test_size(self):
		return self._test_size
	
	@property
	def n_layers(self):
		return self._n_layers
	
	@property
	def n_nodes_per_layer(self):
		return self._n_nodes_per_layer

	@property
	def actvn_per_layer(self):
		return self._actvn_per_layer
	
	@property
	def lr_rate(self):
		return self._lr_rate

	@property
	def optimizer(self):
		return self._optimizer

	@property
	def loss(self):
		return self._loss
	
	@property
	def metrics(self):
		return self._metrics
	
	@property
	def epochs(self):
		return self._epochs

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def parameters(self):
		return "Test size: {}\nLabels: {}\nNo. of layers: {}\nNo. of nodes per layer: {}\nActivation function per layer: {}\nLearning rate: {}\nOptimizer: {}\nLoss: {}\nMetrics: {}\nEpochs: {}\nBatch size: {}".format(self.test_size, self.labels, self.n_layers, self.n_nodes_per_layer, self.actvn_per_layer, self.lr_rate, self.optimizer, self.loss, self.metrics, self.epochs, self.batch_size)
	


	# Setters
	@df.setter
	def df(self, n): #"n" here is a df
		self._df= n

	@labels.setter
	def labels(self, n): #"n" here is a list of those columns in the df you want to use as labels
		self._labels= n

	@test_size.setter
	def test_size(self, n): #"n" here is a float between 0 and 1 with two values after the decimal
		self._test_size= n

	@n_layers.setter
	def n_layers(self, n): #"n" here is a list of integers
		self._n_layers= n

	@n_nodes_per_layer.setter
	def n_nodes_per_layer(self, n): #"n" here is a list of integers
		self._n_nodes_per_layer= n

	@actvn_per_layer.setter
	def actvn_per_layer(self, n): #"n" here is a list of string
		self._actvn_per_layer= n

	@lr_rate.setter
	def lr_rate(self, n): #"n" here is a scalar
		self._lr_rate= n

	@optimizer.setter
	def optimizer(self, n): #"n" here is a string
		self._optimizer= n

	@loss.setter
	def loss(self, n): #"n" here is a string
		self._loss= n

	@metrics.setter
	def metrics(self, n): #"n" here is a list of string
		self._metrics= n

	@epochs.setter
	def epochs(self, n): #"n" here is an integer
		self._epochs= n

	@batch_size.setter
	def batch_size(self, n): #"n" here is an integer
		self._batch_size= n
	

	# API methods
	def parameters(self): #Pass a string with the format: 
		print("Test size: {}\nLabels: {}\nNo. of layers: {}\nNo. of nodes per layer: {}\nActivation function per layer: {}\nLearning rate: {}\nOptimizer: {}\nLoss: {}\nMetrics: {}\nEpochs: {}\nBatch size: {}".format(self.test_size, self.n_layers, self.n_nodes_per_layer, self.actvn_per_layer, self.lr_rate, self.optimizer, self.loss, self.metrics, self.epochs, self.batch_size))

	def normalize(self): # Normalizes DF between 0 and 1
		self.df= ((self.df-self.df.min())/(self.df.max()-self.df.min()))

	def train_test_dataset(self):
		# df with labels: df_Y, df with dataset to train on: df_X
		self.df_y, self.df_X= df[self.labels], df.drop([self.labels], axis= 1)

		# X_train, X_test, y_train, y_test
		self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(self.df_X.values, self.df_y.values, test_size=self.test_size)


	def train(self, plot= False):
		pass