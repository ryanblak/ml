import numpy as np

class LinearRegression:
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
		
		self.cost_history = []
	
	
	def linear_regression(self, type=False):
		"""
		Select the function to use for linear regression, either manually or 
		automatically if nothing is specified by the user.
		
		Gradient descent being an iterative implementation and the normal equation 
		being a non-iterative solution
		"""
		
		if type == 1:
			theta = self.gradient_descent()
		elif type == 2:
			theta = self.normal_equation()
		else:
			n = self.n
			#as n grows, the matrix operations in the normal equation become 
			#expensive. However, it's probably worth reimplenting gradient descent 
			#using a less vectorised implementation, if this is ever a serious 
			#concern.
			if n-1 > 10000:
				theta = self.gradient_descent()
			else:
				theta = self.normal_equation()
		
		self.theta = theta
		return theta
		
	
	def gradient_descent(self, iterations=0):
		"""
		An iterative solution for linear regression. Gradient descent repeatedly 
		compares a hypothesis to the data and adjusts itself until an optimal 
		solution has been reached.
		
		Adjusting alpha will change the rate at which the algorithm will reach a 
		solution. A higher alpha will increase the rate of change, meaning that 
		the algorithm in theory descends quicker, but also increases the chance 
		that it "overshoots" the optimum and diverges away from a solution.
		"""
		
		#assign variables for readability
		n = self.n
		m = self.m
		theta = np.zeros((n, 1))
		alpha = 0.01
		x = self.x
		y = self.y
		
		x = self.feature_normalization(x);
		
		condition = True
		while condition:
			lasttheta = theta
			
			#run the cost function to determine how far from optimal our current 
			#solution is. We store the cost history incase we want to analyse 
			#our learning rate (alpha) and the efficiency of the descent.
			error = ((x * theta) - y)
			squared_error = np.power(error, 2)
			cost = (1/(2*m)) * squared_error.sum()
			self.cost_history.append(cost)
			
			
			#we compare our hypothesis against the data and adjust accordingly.
			diff = (x * theta) - y
			hypothesis = (diff.T * x)
			theta = theta - (alpha/m) * hypothesis.T
			
			#if our theta remains unchanged, it means our gradient is zero and we 
			#have already reached an optimal solution.
			if np.array_equal(lasttheta,theta):
				condition = False
				
		return theta
	
	
	def feature_normalization(self, x):
		"""
		Feature normalization ensures that our gradient descent algorithm 
		converges efficiently.
		
		The mean and standard deviation of each feature is taken. The mean of each 
		feature is subtracted from each instance of a feature and then the result 
		is divided by the standard deviation. This results in our values across 
		all features being within a close range of each other, whilst retaining 
		their magnitude to the other instances of the same feature.
		"""
		
		#we split out our hzero, as this requires no normalization
		m = self.m
		features = self.x[:,1:]
		hzero = self.x[:,0]
		
		mu = features.mean(0)
		sigma = features.std(0)
		
		#generate matrices so that we can apply changes across all points at once
		matrix_mu = np.ones((m, 1))*mu
		matrix_sigma = np.ones((m,1))*sigma
		
		features_norm = (features - matrix_mu) / matrix_sigma
		
		x_norm = np.hstack([hzero,features])
		
		return x_norm
		
	def normal_equation(self):
		"""
		A non iterative solution to linear regression.
		
		The matrix is squared, resulting in an n x n matrix. The inverse is then 
		taken and multiplied by x transposed and y, resulting in an n x 1 matrix 
		and thus a solution to each theta and a complete hypothesis.
		"""
		x = self.x
		y = self.y
		
		square = x.T * x
		theta = square.I * x.T * y
		
		return theta
	
	
	def predict(self, variables):
		"""
		Given an appropriate and complete set of x variables, predict the y 
		variable using the last calculated theta.
		"""
		x = np.matrix([1,variables])
		y = self.theta.T * x.T
		
		return y
	

data = np.genfromtxt('ex1data1.txt', delimiter = ',')
lr = LinearRegression(data)

print(lr.linear_regression())
print(lr.predict(7.1))
