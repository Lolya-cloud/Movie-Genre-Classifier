import torch.nn as nn
import torch


class NN(nn.Module):
    def __init__(self, input_len, output_len):
        print(input_len)
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_len, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.1)
        self.layer4 = nn.Linear(128, 64)
        self.dropout4 = nn.Dropout(0.1)
        self.layer5 = nn.Linear(64, output_len)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = torch.relu(self.layer4(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.layer5(x))
        return x
