import torch
from torch import nn # nn -> neural network
import matplotlib.pyplot as plt

#get pytorch version
torch.__version__

#tensor = torch.rand(size=(3, 3))
#tensor1 = tensor.reshape(9)
#tensor1

weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# Split data into training and testing 80/20
