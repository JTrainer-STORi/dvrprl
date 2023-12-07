import torch
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ParallelEmbeddingLayer(torch.nn.Module):
    """
    Takes the feature vector of each action in the set of actions and
    applies the same fully connected network to each of them.

    Args:
        in_features (int): Size of the input feature vector.
        out_features (int): Size of the output feature vector.
        hidden_layers (list): List of hidden layer sizes.
    """

    def __init__(self, in_features: int, out_features: int, hidden_layers: list[int]) -> None:
        super(ParallelEmbeddingLayer, self).__init__()
        for i in range(len(hidden_layers)):
            if i == 0:
                self.hidden_layers = torch.nn.ModuleList([layer_init(torch.nn.Linear(in_features, hidden_layers[i]))])
            else:
                self.hidden_layers.append(layer_init(torch.nn.Linear(hidden_layers[i - 1], hidden_layers[i])))
        self.output_layer = layer_init(torch.nn.Linear(hidden_layers[-1], out_features))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            input (torch.Tensor): Input feature vector.
        Returns:
            embedding (torch.Tensor): Output feature vector.

        """
        for layer in self.hidden_layers:
            input = torch.nn.functional.tanh(layer(input))
        embedding = torch.nn.functional.tanh(self.output_layer(input))
        return embedding
    
class TransformerLayer(torch.nn.Module):
    """
    Transformer layer.

    Args:
        in_features (int): Size of the input feature vector.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
        hidden_features (int): Size of the hidden layer.
    """

    def __init__(self, in_features: int, num_heads: int, dropout: float, hidden_features: int) -> None:
        super(TransformerLayer, self).__init__()
        self.attention = torch.nn.MultiheadAttention(in_features, num_heads)
        self.feed_forward1 = torch.nn.Linear(in_features, hidden_features)
        self.feed_forward2 = torch.nn.Linear(hidden_features, in_features)
        self.layer_norm1 = torch.nn.LayerNorm(in_features, elementwise_affine=False)
        self.layer_norm2 = torch.nn.LayerNorm(in_features, elementwise_affine=False)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            input (torch.Tensor): Input feature vector.
        Returns:
            embedding (torch.Tensor): Output feature vector.
        """
        attention = self.attention(input, input, input)[0]
        attention = self.dropout1(attention)
        attention = self.layer_norm1(input + attention)
        feed_forward = torch.nn.functional.relu(self.feed_forward1(attention))
        feed_forward = self.feed_forward2(feed_forward)
        feed_forward = self.dropout2(feed_forward)
        embedding = self.layer_norm2(attention + feed_forward)
        return embedding