import numpy as np
from math import e, log
from scipy.optimize import fmin

class LogisticRegression:
	def __init__(self, data):
		"""
		Data passed gets split into x and y. A column of ones is added to 
		represent theta zero. Measurements are taken for future calculations.
		
		Data passed through this function should follow the format of listing 
		all independent variable columns first, followed by a single column 
		containing the dependent variable.
		"""
		
		data_array = np.array(data)
		self.data = data_array
		
		x = np.matrix(data_array[:,:-1])
		#measure the number of observations (rows) we have
		self.m = x.shape[0]
		#generate a column of ones and join with x to represent theta zero
		hzero = np.ones((self.m,1))
		self.x = np.hstack([hzero,x])
		
		y = np.matrix(data_array[:,-1])
		self.y = y.T
		
		#n represents the number of independent variables (columns) + 1
		self.n = self.x.shape[1]
		
		self.initial_theta = self.theta = np.zeros((self.n,1))
	
	def sigmoid(self, z):
		"""
		Given an input of theta*x, the sigmoid function returns the probability 
		that the corresponding y is equal to 1.
		"""
		
		ones = np.ones(z.shape)
		es = ones * e
		return ones/(ones+np.power(es, -z))

	def cost(self, theta=None, x=None, y=None, reg=False):
		"""
		Return a single variable indicating how far from optimal our current 
		values of theta are.
		"""
		#because of the way fmin returns a [n,] matrix, in cases where fmin 
		#provides the theta, we need to ensure that theta is a true [n,1] matrix
		if theta is None:
			theta = self.theta
		else:
			theta = np.matrix(theta)
			theta = theta.T
		if x is None:
			x = self.x
		if y is None:
			y = self.y
		
		m = self.m
		
		#when y=1, y_one is the activated part of the formula. Similarly, when y=0
		#y_zero is the activated part.
		y_one = np.log(self.sigmoid(x*theta))
		y_zero = np.log(1-self.sigmoid(x*theta))
		cost = y_one.T*y + y_zero.T*(1-y)
		
		j = (-(1/m) * cost.sum())
		
		if reg > 0:
			reg_theta = theta
			#ensure that we ignore the bias variable
			reg_theta[0] = 0
			reg_squared = np.power(reg_theta, 2)
			
			j += ((reg/(2*m)) * reg_squared.sum())
		
		return j
	
	def grad(self, theta=None, x=None, y=None, reg=False):
		"""
		Grad computes the partial derivative of the cost function. We use this to 
		determine the gradient of the tangent to the cost function at the 
		provided values of theta. 
		"""
		
		#because of the way fmin returns a [n,] matrix, in cases where fmin 
		#provides the theta, we need to ensure that theta is a true [n,1] matrix
		if theta is None:
			theta = self.theta
		else:
			theta = np.matrix(theta)
			theta = theta.T
		if x is None:
			x = self.x
		if y is None:
			y = self.y
		
		m = self.m
		
		grad = ((1/m) * x.T * (self.sigmoid(x*theta)-y))
		
		if reg > 0:
			reg_theta = theta
			reg_theta[0] = 0
			
			grad += ((reg/m)*reg_theta)
		
		return grad
	
	def map_features(self, degree):
		"""
		For datasets that require a complex non-linear solution, map_features can
		be used to map features to higher order polynomials.
		"""
		
		x1 = np.matrix(self.data[:,1])
		x1 = x1.T
		x2 = np.matrix(self.data[:,2])
		x2 = x2.T
		
		features = self.x[:,1]
		
		for i in range(1,degree+1):
			for j in range(i+1):
				x = np.hstack([features,(np.power(x1, (i-j)) * np.power(x2, j))])
		
		return x
	
	def train(self, reg=False, map_degree=None):
		"""
		Minimize the logistic regression cost function to obtain the optimal 
		values of theta.
		"""
		
		if map_degree is not None:
			x = map_features(map_degree)
		else:
			x = self.x
			
		train_args = (x, self.y, reg)
		
		theta = fmin(self.cost, self.theta, train_args, disp=0)
		
		theta = np.matrix(theta)
		self.theta = theta.T
		
		return self.theta
	
	def predict(self, variables):
		"""
		Given a complete set (or sets) of x inputs, predict returns the likelyhood that the 
		corresponding y value(s) equal 1. Therefore, rounding the output of 
		predict, will give a prediction of which class the supplied variables belong to.
		"""
		
		theta = self.theta
		variables = np.matrix(variables)
		m = variables.shape[0]
		hzero = np.ones((m,1))
		#add a bias variable
		variables = np.hstack([hzero,variables])
		
		p = self.sigmoid(variables*theta)
		
		return p

data = np.genfromtxt('ex2data1.txt', delimiter = ',')
lr = LogisticRegression(data)

print(lr.train())
print(lr.predict([[60,86],[55,36]]))
