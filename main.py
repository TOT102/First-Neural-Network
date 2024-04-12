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
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], X[:train_split]
X_test, y_test = X[train_split:], X[train_split:]

#len(y_train), len(y_test)
# Function -> Plots training data, test data and compares predictions.
def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):

  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
  
  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

  if predictions is not None:
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  # Show the legend
  plt.legend(prop={"size": 14});

plot_predictions();