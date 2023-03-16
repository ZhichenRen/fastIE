import torch
import torch.nn as nn


class BertLayerNorm(nn.Module):
    """This class is LayerNorm model for Bert."""

    def __init__(self, hidden_size, eps=1e-12):
        """This function sets `BertLayerNorm` parameters.

        Arguments:
            hidden_size {int} -- input size

        Keyword Arguments:
            eps {float} -- epsilon (default: {1e-12})
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        """This function propagates forwardly.

        Arguments:
            x {tensor} -- input tesor

        Returns:
            tensor -- LayerNorm outputs
        """

        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertLinear(nn.Module):
    """This class is Linear model for Bert."""

    def __init__(self,
                 input_size,
                 output_size,
                 activation=nn.GELU(),
                 dropout=0.0):
        """This function sets `BertLinear` model parameters.

        Arguments:
            input_size {int} -- input size
            output_size {int} -- output size

        Keyword Arguments:
            activation {function} -- activation function (default: {gelu})
            dropout {float} -- dropout rate (default: {0.0})
        """

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.linear.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.bias.data.zero_()
        self.activation = activation
        self.layer_norm = BertLayerNorm(self.output_size)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def get_input_dims(self):
        return self.input_size

    def get_output_dims(self):
        return self.output_size

    def forward(self, x):
        """This function propagates forwardly.

        Arguments:
            x {tensor} -- input tensor

        Returns:
            tenor -- Linear outputs
        """

        output = self.activation(self.linear(x))
        return self.dropout(self.layer_norm(output))
