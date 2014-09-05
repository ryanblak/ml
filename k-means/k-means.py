from __future__ import division
import numpy as np
from PIL import Image


class KMeans:
	def __init__(self, data):
		"""
		Ensure that data is a numpy array and take measures of the data.
		"""
		
		self.data = np.array(data)
		self.m = data.shape[0]
		self.n = data.shape[1]
	
	
	def initialise_centroids(self, data, num_centroids):
		"""
		Randomly select data points to act as the initial centroid values.
		"""
		
		np.random.permutation(data)
		
		return data[:num_centroids,:]
	
	
	def cluster(self, centroids):
		"""
		Measure the distance from each centroid and select the closest centroid 
		for each of the data points.
		"""
		
		#reshape and duplicate data across the 3rd dimension		
		cube_data = np.reshape(self.data, (1, self.m, self.n))
		cube_data = np.tile(cube_data, (centroids.shape[0], 1, 1))
		
		#reshape across the 3rd dimension and duplicate centroids across the 1st 
		#dimension
		cube_centroids = np.reshape(centroids, (centroids.shape[0], 1, self.n))
		cube_centroids = np.tile(cube_centroids, (1, self.m, 1))
		
		# measure the x and y squared distances and sum to compute the euclidian 
		#distance and return the index id of the closest centroid.
		distancesq = np.power(cube_data - cube_centroids, 2)
		index = np.argmin(distancesq.sum(axis=2), axis=0)
		
		index = np.reshape(index, (self.m, 1))
		
		return index.astype(int)
	
	
	def calculate(self, index):
		"""
		Recompute centroids so that they are at the centred based on the members 
		of their group.
		"""
		
		m_index = np.matrix(index)
		#construct a logical matrix from numerical index list
		bin_index = np.matrix(np.eye(np.amax(index)+1)[index,:]).T
		
		#get a sum of all the points x and y values
		cluster_sum = bin_index * self.data
		#get a count of the number of clusters per matrix
		cluster_count = np.sum(bin_index, axis=1)
		
		#compute the mean position of each cluster
		centroids = np.divide(cluster_sum, cluster_count)
		
		return np.array(centroids)
	
	
	def train(self, num_centroids):
		"""
		Continually assign data points to their closest centroid and recompute 
		centroids until the centroids reach an optimal solution and no longer 
		change position when recomputed.
		"""
		
		centroids = self.initialise_centroids(self.data, num_centroids)
		
		optimizing = True
		
		while optimizing:
			prev_centroids = centroids
			
			index = self.cluster(centroids)
			centroids = self.calculate(index)
			
			if(np.array_equal(centroids, prev_centroids)):
				optimizing = False
		
		return centroids, index


im = Image.open('bird_small.png','r')
data = np.asarray(im.getdata())
width, height = im.size

km = KMeans(data)
#train the algorithm with the desired number of colours
centroids, index = km.train(16)

c_data = centroids[index,:]
c_data = np.squeeze(c_data)
c_data = np.reshape(c_data, (height, width, 3))
c_img = Image.fromarray(np.uint8(c_data))
c_img.show()

