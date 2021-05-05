import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# image = torch.randn((1, 1, 5, 5, 5), dtype=torch.float)
image = [[[[
[0.13, 0.13, 0.98, -0.50, 0.26],
[1.74, -2.55, -0.60, -0.81, -1.35],
[1.21, -0.29, 0.44, -0.87, -0.73],
[-1.46, 0.12, -0.65, 0.86, 0.84],
[-1.08, -0.21, 0.54, -0.31, -1.04],
],

[[0.07, 0.53, 0.02, 0.54, -0.88],
[0.67, -0.29, 0.77, -1.19, 0.90],
[2.29, 1.43, -1.06, 0.09, -0.09],
[-1.23, 0.78, 1.32, 1.10, -0.17],
[-0.18, 0.41, 0.72, 0.56, -0.42],
],

[[0.57, -0.54, 0.23, 0.71, -1.83],
[0.94, -0.07, -0.18, -0.77, 2.35],
[0.42, -0.87, 2.12, -0.36, -0.06],
[1.02, 0.07, 0.20, 0.59, 0.82],
[-0.98, -0.41, 1.22, -1.95, 0.28],],

[[1.30, 1.22, 0.47, -1.32, -0.15],
[0.03, 0.71, 0.60, 0.08, 0.99],
[-0.81, -0.14, -0.31, 1.80, -0.90],
[-0.38, 0.28, -1.25, 1.38, 1.75],
[-0.62, 1.69, -0.94, 0.49, 0.56],],

[[0.65, 0.43, 0.36, 0.62, -1.44],
[-0.30, 0.24, -1.04, -0.21, -0.97],
[-0.12, 0.81, 1.17, -0.01, -0.41],
[0.12, 1.37, 0.37, -0.62, 1.33],
[-0.04, -2.55, -1.93, 0.21, 1.65],]]]]
image = torch.tensor(image, dtype=torch.float)

layer = nn.AvgPool3d((3,3,3), stride=(1,1,1), padding=(0,0,0))

# Get output
output = layer(image)

# From device to cpu and numpy
a = image.cpu().detach().numpy()
b = output.cpu().detach().numpy()

# Print tensors
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print("Input:")
print(a)
print("-------------------")
print("Output:")
print(b)
