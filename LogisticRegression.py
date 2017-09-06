import numpy as np
import math
import time

class LogisticRegression:
	def __init__(self):
		self.weights = []

	def fit(self, X, y):
		X = np.c_[np.ones((X.shape[0], 1)), X]
		self.weights = np.ones((X.shape[1], 1))
		self.MBGD(X, y)

	def predict(self, x):
		#s = self.weights.T.dot(x)[0]
		s = x.dot(self.weights)
		return LogisticRegression.sigma(s)

	def log_loss(self, X, y):
		sum = 0
		for i in range(X.shape[0]):
			p = self.predict(X[i])
			yi = y[i]
			sum += yi * math.log(p) + (1 - yi) * math.log(1 - p)
		return -sum / X.shape[0]

	def gradients(self, X, y):
		grads = np.zeros((X.shape[1], 1))
		for j in range(self.weights.shape[0]):
			sum = 0
			predicted =self.predict(X)
			dif = (predicted-y)
			feat = X[:,j][np.newaxis]
			s = (predicted-y)*feat
			#for i in range(X.shape[0]):
			#	sum += (self.predict(X[i]) - y[i][0]) * X[i][j]
			grads[j] = sum / X.shape[0]
		return grads

	def BGD(self, X, y):
		learning_rate = 0.75
		n_iterations = 7
		for n_iterations in range(n_iterations):
			print("iteration number:")
			print(n_iterations)
			self.weights = self.weights - learning_rate * self.gradients(X, y)
			print("weights:")
			print(self.weights)
			print("loss:")
			print(self.log_loss(X, y))

	def MBGD(self, X, y):
		learning_rate = 0.5
		n_iterations = 100
		batch_size = 0.02

		if batch_size < 1:
			batch_size = int(X.shape[0] * batch_size)

		seed = np.random.randint(0, high=100)
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)

		for n_iterations in range(n_iterations):
			print("iteration number:")
			print(n_iterations)
			self.weights = self.weights - learning_rate * self.gradients(X[:batch_size], y[:batch_size])
			print("weights:")
			print(self.weights)
			print("loss:")
			print(self.log_loss(X, y))

	def SGD(self, X, y):
		learning_rate = 0.2
		n_iterations = 100000
		n_instance = np.random.randint(0, X.shape[0]+1)
		for n_iterations in range(n_iterations):
			print("iteration number:")
			print(n_iterations)
			self.weights = self.weights - learning_rate * self.gradients(X[n_instance:n_instance+1], y[n_instance:n_instance+1])
		print("weights:")
		print(self.weights)
		print("loss:")
		print(self.log_loss(X, y))

	@classmethod
	def sigma(cls, x):
		res = 1 / (1 + np.e ** (-x))
		res[res == 1] = 0.9999999999999
		res[res == 0] = 0.0000000000001
		#if res == 1:
		#	res = 0.9999999999999
		#if res == 0:
		#	res = 0.0000000000001
		return res
