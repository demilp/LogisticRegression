import numpy as np
import math
import time

class LogisticRegression:
	def __init__(self):
		self.weights = None

	def fit(self, X, y):
		X = np.c_[np.ones((X.shape[0], 1)), X]
		self.weights = np.ones((1, X.shape[1]))
		self.MBGD(X, y)

	def predict(self, x):
		#s = self.weights.T.dot(x)[0]
		s = x.dot(self.weights.T)
		return LogisticRegression.sigma(s)

	def log_loss(self, X, y):
		p = self.predict(X)
		return -np.mean(y * np.log(p) + (1-y) * np.log(1 - p))

	def gradients(self, X, y):
		grads = np.zeros((X.shape[1], 1))
		for j in range(self.weights.shape[1]):
			#p = self.predict(X)
			#f = X[:, j][np.newaxis].T
			#grads[j] = np.mean((p - y) * f)
			grads[j] = np.mean((self.predict(X)-y)*(X[:, j][np.newaxis]).T)
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
		print("loss:")
		print(self.log_loss(X, y))
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
			self.weights = self.weights - learning_rate * self.gradients(X[:batch_size], y[:batch_size])
			if n_iterations % 5 == 0:
				print("iteration number:")
				print(n_iterations)
				print("loss:")
				print(self.log_loss(X, y))

	def SGD(self, X, y):
		learning_rate = 0.2
		n_iterations = 20000
		n_instance = np.random.randint(0, X.shape[0]+1)
		for n_iterations in range(n_iterations):

			self.weights = self.weights - learning_rate * self.gradients(X[n_instance:n_instance+1], y[n_instance:n_instance+1])
			if n_iterations % 100 == 0:
				print("iteration number:")
				print(n_iterations)
				print("loss:")
				print(self.log_loss(X, y))
		print("weights:")
		print(self.weights)


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
