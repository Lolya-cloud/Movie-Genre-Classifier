import torch.nn as nn
import torch


class NN(nn.Module):
    def __init__(self, input_len, output_len):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_len, 256)
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(256, output_len)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.layer2(x))
        return x