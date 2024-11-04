# Assuming you have already imported the necessary libraries
import numpy as np
from parameters import *
from utils import data
from model import NeuralNetwork
import matplotlib.pyplot as plt

data = data()
nn = NeuralNetwork()
nn.init_weights()
nn.train(data)
