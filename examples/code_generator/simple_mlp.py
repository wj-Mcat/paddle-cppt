import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        self.fc = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.final_fc = nn.Linear(in_features=hidden_size, out_features=hidden_size)
    
    def forward(self, embedding):
        output = self.fc(embedding)
        return self.final_fc(output)
def run(input_size: int, hidden_size: int):
    model = MLP(input_size, hidden_size)
    inputs = torch.randn(1, input_size)
    outputs = model(inputs)
    return outputs.shape
