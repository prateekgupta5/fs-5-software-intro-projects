import torch
from torch import nn

# Create a Linear Regression model class
class PID_Aproximator(nn.Module): # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
  def __init__(self):
    super().__init__() 

    #use linear layer to readably and automatically make linear model
    self.linear_layer = nn.Linear(in_features = 2, out_features = 1)


  # Forward defines the computation in the model
  def forward(self, x: torch.Tensor) -> torch.Tensor: # <- "x" is the input data (e.g. training/testing features)
    return self.linear_layer(x)