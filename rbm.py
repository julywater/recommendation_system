import numpy as np
import tensorflow as tf

class RBM(object):
	def __init__(self, visibleDimensions, hiddenDimensions=50, ratingValues=10, learningRate=0.001):
		self.visibleDimensions = visibleDimensions
        self.hiddenDimensions = hiddenDimensions
        self.ratingValues = ratingValues
        self.learningRate = learningRate
		self.X = tf.placeholder(tf.float32, [None, self.visibleDimensions], name="X")
	def init_parameters(self):
		maxWeight = -4.0 * np.sqrt(6.0 / (self.hiddenDimensions + self.visibleDimensions))
        self.weights = tf.Variable(tf.random_uniform([self.visibleDimensions, self.hiddenDimensions], minval=-maxWeight, maxval=maxWeight), tf.float32, name="weights")
        self.hBias = tf.Variable(tf.zeros([self.hiddenDimensions], tf.float32,name="hBias"))
        self.vBias = tf.Variable(tf.zeros([self.visibleDimensions],tf.float32,name="vBias"))
    def gibbs_sampling(self):  
