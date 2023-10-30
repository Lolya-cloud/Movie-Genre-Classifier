import torch.nn as nn


class FlexibleNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, dropout_rates, hidden_layer_activation, output_layer_activation):
        super(FlexibleNeuralNetwork, self).__init__()
        if len(layer_sizes) != len(dropout_rates) + 1:
            raise ValueError("number of dropout rates should be one less than the number of layers")
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # we are creating linear layers, so the len is n-1
        for i in range(len(layer_sizes) - 1):
            self.layers.append((nn.Linear(layer_sizes[i], layer_sizes[i + 1])))
            # add dropouts to every layer except the last one.
            if i < len(dropout_rates):
                self.dropouts.append(nn.Dropout(dropout_rates[i]))

    def forward(self, x):
        # activation for each layer
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i < len(self.layers) - 1:
                x = self.hidden_layer_activation(x)
                x = self.dropouts[i](x)
            else:
                x = self.output_layer_activation(x)
        return x
