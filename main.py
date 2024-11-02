import numpy as np
from parameters import *
from utils import data
from model import NeuralNetwork


data = data()
nn = NeuralNetwork()
nn.init_weights()

print(nn.train(data))
