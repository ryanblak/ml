from __future__ import division
import numpy as np
import scipy.io as sio
from math import log
from scipy.optimize import minimize

class NeuralNetwork:
	def __init__(self, data, hidden_layer):
		"""
		Data passed gets split into x and y. A column of ones is added to 
		act as the bias variable. Measurements are taken for future calculations.
		
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
		
		#use hard coded layers for this project
		self.input_layer_size = x.shape[1]
		self.hidden_layer_size = hidden_layer
		self.output_layer_size = np.amax(y)+1
		
		#initialise theta with random weights to ensure that forward propagation 
		#functions correctly 
		self.initial_theta1 = self.theta1 = self.rand_initial_weights(\
			self.hidden_layer_size, (self.input_layer_size+1))
		
		self.initial_theta2 = self.theta2 = self.rand_initial_weights(\
			self.output_layer_size, (self.hidden_layer_size+1))
		
		#unroll parameters for use in fmin
		self.initial_nn_params = self.nn_params = np.hstack(\
		[np.ravel(self.theta1),np.ravel(self.theta2)])
	
	
	def cost(self, nn_params=None, x=None, y=None, reg=False):
		"""
		Return the cost and theta gradients. 
		The cost is a single numerical value calculated by performing 
		forward-propogation. 
		The theta gradients are an unrolled array equivalent in size to theta1 and 
		theta2 obtained through performing back-propogation.
		"""
		
		if nn_params is None:
			nn_params = self.nn_params
		if x is None:
			x = self.x
		if y is None:
			y = self.y
		
		m = self.m
		
		#convert the y column into a logical matrix
		vector_y = np.matrix(np.eye(self.output_layer_size)[y.astype(int),:])
		
		#reshape and split nn_params from a one dimensional array of all theta 
		#values, into the matrix form of theta1 and theta2
		theta1, theta2 = self.roll_theta(nn_params)
		
		#perform forward propagation
		a1 = x;
		z2 = a1*theta1.T
		a2 = np.hstack([np.ones((m,1)), self.sigmoid(z2)])
		z3 = a2*theta2.T
		a3 = self.sigmoid(z3);
		
		#perform log operations outside of cost function to replace any -inf
		loga3 = np.log(a3)
		loga3[loga3==-np.inf]=0
		log1a3 = np.log(1-a3)
		log1a3[log1a3==-np.inf]=0
		
		j = -(1/m) * np.sum(np.multiply(loga3, vector_y) + np.multiply(log1a3,\
			1-vector_y))
		
		if reg > 0:
			reg_theta1 = theta1
			reg_theta2 = theta2
			#ensure that we ignore the bias variables
			reg_theta1[:,0] = 0
			reg_theta2[:,0] = 0
			
			j += (reg/(2*m)) * (np.sum(np.power(reg_theta1,2)) +\
				np.sum(np.power(reg_theta2,2)))
		
		#perform back propagation
		d3 = a3 - vector_y
		d2 = np.multiply((d3 * theta2[:,1:]), self.sigmoid_grad(z2))
		
		D2 = d3.T * a2
		D1 = d2.T * a1
		
		theta1_grad = (1/m)*D1
		theta2_grad = (1/m)*D2
		
		if reg > 0:
			reg_theta1 = theta1
			reg_theta2 = theta2
			#ensure that we ignore the bias variables
			reg_theta1[:,0] = 0
			reg_theta2[:,0] = 0
			
			theta1_grad += (reg/m)*reg_theta1
			theta2_grad += (reg/m)*reg_theta2
			
		#unroll theta_grads into a 1D array
		nn_params = np.hstack([np.ravel(theta1_grad, order="F"),\
			np.ravel(theta2_grad, order="F")])
		
		return j, nn_params.T
			
	
	def sigmoid(self, z):
		"""
		Given an input of theta*x, the sigmoid function returns the probability 
		that the corresponding y is equal to 1.
		"""
		
		return 1/(1+np.exp(-z))
	
	
	def sigmoid_grad(self, z):
		"""
		The sigmoid gradient function returns the partial derivate of the sigmoid
		function.
		"""
		
		return np.multiply(1/(1+np.exp(-z)),1-(1/(1+np.exp(-z))))
	
	
	def rand_initial_weights(self, l_in, l_out):
		"""
		Return a l_in by l_out matrix of small random variables. 
		"""
		
		epsilon_init = 0.12
		return np.random.rand(l_out, l_in) * 2 * epsilon_init - epsilon_init
	
	
	def roll_theta(self, nn_params):
		"""
		Take an unrolled array of theta and roll them back into their matrix 
		shapes.
		"""
		
		theta1 = np.matrix(np.reshape(nn_params[:((self.input_layer_size+1)*\
			self.hidden_layer_size)],(self.hidden_layer_size,\
			(self.input_layer_size+1)),order="F"))
		theta2 = np.matrix(np.reshape(nn_params[((self.input_layer_size+1)*\
			self.hidden_layer_size):],(self.output_layer_size,\
			(self.hidden_layer_size+1)),order="F"))
		
		return theta1, theta2
	
	
	def train(self, reg=False):
		"""
		Train the neural network by minimizing the cost function. Store and return
		the optimal values of theta.
		"""
		
		train_args = (self.x, self.y, reg)
		train_options = {'maxiter':400}
		
		nn_params = minimize(self.cost, self.nn_params, train_args, method="CG",\
			jac=True, options=train_options)
		
		nn_params = nn_params['x']
		
		self.theta1, self.theta2 = self.roll_theta(nn_params)
		self.nn_params = nn_params
		
		return self.nn_params
	
	
	def predict(self, variables):
		"""
		Given a set of x variables, return predictions for the corresponding y 
		variables.
		"""
		
		m = self.m
		
		h1 = self.sigmoid(np.matrix(np.hstack([np.ones((m,1)),variables]))*\
			self.theta1.T)
		h2 = self.sigmoid(np.matrix(np.hstack([np.ones((m,1)),h1]))*self.theta2.T)
		
		return np.argmax(h2, axis=1)


#load in MATLAB formatted data
data = sio.loadmat('ex4data1.mat')
data_x = data['X']
data_y = data['y']
#remap 10 to 0, as MATLAB/Octave uses one-based arrays
data_y[data_y > 9] = 0

data = np.hstack([data_x,data_y])
data = data.astype(int)
nn = NeuralNetwork(data, 25)

nn.train()
#evaluate predictions based on how well they perform on the training set, though
#splitting out a test set and evaluating on that would be a better test of 
#accuracy
predictions = np.matrix(nn.predict(data[:,:-1]))
print('Accuracy of {}% on training set.'.format((np.mean((predictions ==\
	data[:,-1:]).astype(int)))*100))


