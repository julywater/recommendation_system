import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

class RBM(object):

        def __init__(self, visibleDimensions, epochs=20, hiddenDimensions=50, ratingValues=10, learningRate=0.001, batchSize=100):

                    self.visibleDimensions = visibleDimensions
                    self.epochs = epochs
                    self.hiddenDimensions = hiddenDimensions
                    self.ratingValues = ratingValues
                    self.learningRate = learningRate
                    self.batchSize = batchSize
