from __future__ import division
import numpy as np
from PIL import Image


class KMeans:
	def __init__(self, data):
		"""
		Ensure that data is a numpy matrix and take measures of the data.
		"""
		
		self.data = np.matrix(data)
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
		
		index = np.zeros((self.m,1))
		
		for i in range(self.m):
			distancesq = np.power((np.tile(self.data[i,:],(centroids.shape[0], 1))\
				- centroids),2)
			index[i] = np.argmin(distancesq.sum(axis=1))
		
		return index.astype(int)
	
	
	def calculate(self, index):
		"""
		Recompute centroids so that they are at the centred based on the members 
		of their group.
		"""
		
		centroids = np.zeros((np.amax(index)+1,self.n))
		
		for i in range(np.amax(index)+1):
			
			centroids[i,:] = (1/np.sum(index == i))\
				*np.sum(self.data[np.where(index == i)[0]], axis=0)
		
		return centroids
	
	
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
c_img.show

