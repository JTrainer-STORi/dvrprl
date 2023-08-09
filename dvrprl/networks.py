import torch
from einops.layers.torch import Reduce

from dvrprl.layers import ParallelEmbeddingLayer, TransformerLayer


class ParallelMultilayerPerceptron(torch.nn.Module):
    """
    Parallel Multilayer Perceptron. This network applies the same fully connected
    network to each of the items in the set. The output of the network
    is then either the value of the state, the action-value of the state-action pair
    or the probability of each action depending on the mode argument.

    Args:
        in_features (int): Size of the input feature vector for each item.

        embedding_dim (int): Size of the output feature vector for each item.

        hidden_layers (list): List of hidden layer sizes.

        mode (str): Either "value", "action-value" or "policy" depending on the
        desired output of the network.
    """

    def __init__(
        self, in_features: int, embedding_dim: int, hidden_layers: list[int], mode: str
    ) -> None:
        super(ParallelMultilayerPerceptron, self).__init__()
        self.embedding_layer = ParallelEmbeddingLayer(
            in_features, embedding_dim, hidden_layers
        )
        if mode == "value":
            self.final_stack = torch.nn.ModuleList(
                modules=[
                    torch.nn.Linear(in_features=embedding_dim, out_features=1),
                    Reduce(pattern="b n d -> b d", reduction="sum"),
                ]
            )
        elif mode == "action-value":
            self.final_stack = torch.nn.ModuleList(
                modules=[torch.nn.Linear(in_features=embedding_dim, out_features=1)]
            )
        elif mode == "policy":
            self.final_stack = torch.nn.ModuleList(
                [
                    torch.nn.Linear(in_features=embedding_dim, out_features=1),
                    torch.nn.ReLU(),
                    torch.nn.Softmax(dim=1),
                ]
            )

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
    """
    Transformer network. This network consists of an embedding layer that is applied
    to each item in the input set. This embedded set is then fed to a transformer layer
    which applies multiheaded attention to it. The output of the network
    is then either the value of the state, the action-value of the state-action pair
    or the probability of each action depending on the mode argument.

    Args:
        in_features (int): Size of the input feature vector for each item.
        embedding_dim (int): Size of the output feature vector from the embedding layer for each item.
        hidden_layers (list): List of hidden layer sizes.
        hidden_dim (int): Size of the hidden dimension in the transformer layer.
        num_heads (int): Number of heads in the multiheaded attention.
        dropout (float): Dropout rate.
        num_blocks (int): Number of transformer blocks.
        mode (str): Either "value", "action-value" or "policy" depending on the
        desired output of the network.

    """

    def __init__(
        self,
        in_features: int,
        embedding_dim: int,
        hidden_layers: list[int],
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_blocks: int,
        mode: str,
    ) -> None:
        super(Transformer, self).__init__()
        self.embedding_layer = ParallelEmbeddingLayer(
            in_features, embedding_dim, hidden_layers
        )
        self.transformer_layers = torch.nn.ModuleList(
            [
                TransformerLayer(embedding_dim, num_heads, dropout, hidden_dim)
                for _ in range(num_blocks)
            ]
        )
        if mode == "value":
            self.final_stack = torch.nn.ModuleList(
                [
                    torch.nn.Linear(in_features=embedding_dim, out_features=1),
                    Reduce(pattern="b n d -> b d", reduction="sum"),
                ]
            )
        elif mode == "action-value":
            self.final_stack = torch.nn.ModuleList(
                [torch.nn.Linear(in_features=embedding_dim, out_features=1)]
            )
        elif mode == "policy":
            self.final_stack = torch.nn.ModuleList(
                [
                    torch.nn.Linear(in_features=embedding_dim, out_features=1),
                    torch.nn.ReLU(),
                    torch.nn.Softmax(dim=1),
                ]
            )

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
    """
    Recurrent neural network. This network consists of an embedding layer that is applied
    to each item in the input set. This embedded set is then fed to a recurrent layer
    which applies either LSTM or GRU to it. The output of the network
    is then either the value of the state, the action-value of the state-action pair
    or the probability of each action depending on the mode argument.

    Args:
        in_features (int): Size of the input feature vector for each item.
        embedding_dim (int): Size of the output feature vector from the embedding layer for each item.
        hidden_layers (list): List of hidden layer sizes.
        hidden_features (int): Size of the hidden dimension in the recurrent layer.
        unit (str): Either "LSTM" or "GRU" depending on the desired recurrent unit.
        bidirectional (bool): Whether the recurrent layer is bidirectional or not.
        mode (str): Either "value", "action-value" or "policy" depending on the
        desired output of the network.
    """

    def __init__(
        self,
        in_features: int,
        embedding_dim: int,
        hidden_layers: list[int],
        hidden_features: int,
        unit: str,
        bidirectional: bool,
        mode: str,
    ) -> None:
        super(RecurrentNeuralNetwork, self).__init__()
        self.embedding_layer = ParallelEmbeddingLayer(
            in_features, embedding_dim, hidden_layers
        )
        if unit == "LSTM":
            self.rnn = torch.nn.LSTM(
                embedding_dim,
                hidden_features,
                bidirectional=bidirectional,
                batch_first=True,
            )
        elif unit == "GRU":
            self.rnn = torch.nn.GRU(
                embedding_dim,
                hidden_features,
                bidirectional=bidirectional,
                batch_first=True,
            )
        if mode == "value":
            self.final_stack = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=hidden_features * 2
                        if bidirectional
                        else hidden_features,
                        out_features=1,
                    ),
                    Reduce(pattern="b n d -> b d", reduction="sum"),
                ]
            )
        elif mode == "action-value":
            self.final_stack = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=hidden_features * 2
                        if bidirectional
                        else hidden_features,
                        out_features=1,
                    )
                ]
            )
        elif mode == "policy":
            self.final_stack = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=hidden_features * 2
                        if bidirectional
                        else hidden_features,
                        out_features=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Softmax(dim=1),
                ]
            )

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
