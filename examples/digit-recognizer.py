# import numpy as np
# import pandas as pd

from perceptron import Layer, MultiLayerPerceptron

layer_1 = Layer(16)
layer_2 = Layer(16)
layer_out = Layer(10)

model = MultiLayerPerceptron([layer_1, layer_2, layer_out])
