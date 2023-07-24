import torch
from dvrprl.layers import ParallelEmbeddingLayer, TransformerLayer
from einops.layers.torch import Reduce
    
class ParallelMultilayerPerceptron(torch.nn.Module):
    """
    Parallel Multilayer Perceptron. This network applies the same fully connected
    network to each of the actions in the set of actions. The output of the network
    is then either the value of the state, the action-value of the state-action pair
    or the probability of each action depending on the mode argument.

    Args:
        in_features (int): Size of the input feature vector for each action.

        embedding_dim (int): Size of the output feature vector for each action.

        hidden_layers (list): List of hidden layer sizes.

        mode (str): Either "value", "action-value" or "policy" depending on the
        desired output of the network.
    """

    def __init__(self, in_features: int, embedding_dim: int, hidden_layers: list[int], mode: str) -> None:
        super(ParallelMultilayerPerceptron, self).__init__()
        self.embedding_layer = ParallelEmbeddingLayer(in_features, embedding_dim, hidden_layers)
        if mode == "value":
            self.final_stack = torch.nn.ModuleList(modules=[
                torch.nn.Linear(in_features=embedding_dim, out_features=1),
                Reduce(pattern='b n d -> b d', reduction='sum')
            ])
        elif mode == "action-value":
            self.final_stack = torch.nn.ModuleList(modules=[
                torch.nn.Linear(in_features=embedding_dim, out_features=1)
            ])
        elif mode == "policy":
            self.final_stack = torch.nn.ModuleList([
                torch.nn.Linear(in_features=embedding_dim, out_features=1),
                torch.nn.ReLU(),
                torch.nn.Softmax(dim=1)
            ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the network.

        Args:
            input (torch.Tensor): Feature vector of each node in the graph. 

            shape (batch_size, num_nodes, in_features)
        Returns:
            Depending on the mode, the output is either the value of the state,
            shape (batch_size, 1), the value of the actions, shape (batch_size, num_actions, 1),
            the probability of each action shape (batch_size, num_actions, 1).

        """
        embedding = self.embedding_layer(input)
        for layer in self.final_stack:
            embedding = layer(embedding)
        return embedding
    
class Transformer(torch.nn.Module):

    def __init__(self, in_features: int, embedding_dim: int, hidden_layers: list[int], hidden_dim: int, num_heads: int, dropout: float, num_blocks: int, mode: str) -> None:
        super(Transformer, self).__init__()
        self.embedding_layer = ParallelEmbeddingLayer(in_features, embedding_dim, hidden_layers)
        self.transformer_layers = torch.nn.ModuleList([TransformerLayer(embedding_dim, num_heads, dropout, hidden_dim) for _ in range(num_blocks)])
        if mode == "value":
            self.final_stack = torch.nn.ModuleList([
                torch.nn.Linear(in_features=embedding_dim, out_features=1),
                Reduce(pattern='b n d -> b d', reduction='sum')
            ])
        elif mode == "action-value":
            self.final_stack = torch.nn.ModuleList([
                torch.nn.Linear(in_features=embedding_dim, out_features=1)
            ])
        elif mode == "policy":
            self.final_stack = torch.nn.ModuleList([
                torch.nn.Linear(in_features=embedding_dim, out_features=1),
                torch.nn.ReLU(),
                torch.nn.Softmax(dim=1)
            ])
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the network.

        Args:
            input (torch.Tensor): Feature vector of each node in the graph. 

            shape (batch_size, num_nodes, in_features)
        Returns:
            Depending on the mode, the output is either the value of the state,
            shape (batch_size, 1), the value of the actions, shape (batch_size, num_actions, 1),
            or the probability of each action shape (batch_size, num_actions, 1).
        """

        embedding = self.embedding_layer(input)
        for transformer_layer in self.transformer_layers:
            embedding = transformer_layer(embedding)
        for layer in self.final_stack:
            embedding = layer(embedding)
        return embedding

class RecurrentNeuralNetwork(torch.nn.Module):

    def __init__(self, in_features: int, embedding_dim: int, hidden_layers: list[int], hidden_features: int, unit: str, bidirectional: bool, mode: str) -> None:
        super(RecurrentNeuralNetwork, self).__init__()
        self.embedding_layer = ParallelEmbeddingLayer(in_features, embedding_dim, hidden_layers)
        if unit == "LSTM":
            self.rnn = torch.nn.LSTM(embedding_dim, hidden_features, bidirectional=bidirectional, batch_first=True)
        elif unit == "GRU":
            self.rnn = torch.nn.GRU(embedding_dim, hidden_features, bidirectional=bidirectional, batch_first=True)
        if mode == "value":
            self.final_stack = torch.nn.ModuleList([
                torch.nn.Linear(in_features=hidden_features*2 if bidirectional else hidden_features, out_features=1),
                Reduce(pattern='b n d -> b d', reduction='sum')
            ])
        elif mode == "action-value":
            self.final_stack = torch.nn.ModuleList([
                torch.nn.Linear(in_features=hidden_features*2 if bidirectional else hidden_features, out_features=1)
            ])
        elif mode == "policy":
            self.final_stack = torch.nn.ModuleList([
                torch.nn.Linear(in_features=hidden_features*2 if bidirectional else hidden_features, out_features=1),
                torch.nn.ReLU(),
                torch.nn.Softmax(dim=1)
            ])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the network.

        Args:
            input (torch.Tensor): Feature vector of each node in the graph. 

            shape (batch_size, num_nodes, in_features)
        Returns:
            Depending on the mode, the output is either the value of the state,
            shape (batch_size, 1), the value of the actions, shape (batch_size, num_actions, 1),
            or the probability of each action shape (batch_size, num_actions, 1.
        """
        embedding = self.embedding_layer(input)
        embedding, _ = self.rnn(embedding)
        for layer in self.final_stack:
            embedding = layer(embedding)
        return embedding
