
import torch
from torch import nn # nn -> neural network
import matplotlib.pyplot as plt

# Зареждане на модел

loaded_model_0 = LinearRegressionModel() 

MODEL_SAVE_PATH = "models/01_first_model_basic_workflow.pth"

# Load the saved state dict -> loaded_model_0

loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# 1. Put the loaded model into evaluation mode
loaded_model_0.eval()

# 2. Use the inference mode context manager to make predictions
data = torch.tensor([[0.8000],
                    [0.8200],
                    [0.8400],
                    [0.8600],
                    [0.8800],
                    [0.9000],
                    [0.9200],
                    [0.9400],
                    [0.9600],
                    [0.9800]])

with torch.inference_mode():
    loaded_model_preds = loaded_model_0(data) # perform a forward pass on the test data with the loaded model

plot_predictions 