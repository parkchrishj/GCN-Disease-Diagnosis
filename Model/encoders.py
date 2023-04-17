import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from Model.mlp import MLP


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    implementation of the GraphSage encoder, a "convolutional" approach for encoding node features in graph neural networks. 
    It takes in node features, feature dimensions, embedding dimensions, adjacency lists, aggregator, and other parameters 
    as input. It uses multi-layer perceptrons (MLPs) with batch normalization and ReLU activation functions to process the 
    node features and generate embeddings.
    """

    def __init__(self, features, feature_dim,
                 embed_dim, adj_lists, aggregator,
                 num_sample=10,
                 base_model=None, gcn=False, cuda=False,
                 kernel="GCN",
                 feature_transform=False):
        """
        Constructor for the Encoder class.

        Args:
        - features (nn.Module): Feature module that maps node ids to their corresponding feature representations
        - feature_dim (int): Dimension of input feature embeddings
        - embed_dim (int): Dimension of output embeddings for nodes
        - adj_lists (dictionary): Dictionary containing adjacency lists for each node in the graph
        - aggregator (nn.Module): Aggregator module that performs neighborhood aggregation
        - num_sample (int): Number of neighbors to sample during aggregation (default: 10)
        - base_model (nn.Module): Base model to be used for aggregation (default: None)
        - gcn (bool): Whether to use Graph Convolutional Networks (GCN) for aggregation (default: False)
        - cuda (bool): Whether to use GPU for computation (default: False)
        - kernel (str): Type of kernel to use for aggregation (options: "GCN", "GIN", "GAT") (default: "GCN")
        - feature_transform (bool): Whether to apply feature transformation (default: False)
        """
        super(Encoder, self).__init__()

        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

        self.kernel = kernel

        num_mlp_layers = 2
        num_layers = 5
        self.num_layers = num_layers

        input_dim = feature_dim
        hidden_dim = 512
        output_dim = embed_dim
        self.mlps = torch.nn.ModuleList()
        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(
                    MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(
                    MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        self.mlps.append(
                    MLP(num_mlp_layers, hidden_dim, hidden_dim, embed_dim))
        self.batch_norms.append(nn.BatchNorm1d(embed_dim))

        # Linear function that maps the hidden representation at dofferemt layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        # print('encoder:', nodes)
        if self.kernel == "GIN":
            neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                                  self.num_sample, average="sum")
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            h = torch.add(self_feats, neigh_feats)
            for layer in range(self.num_layers):
                pooled_rep = self.mlps[layer](h)
                h = self.batch_norms[layer](pooled_rep)
                h = F.relu(h)
            combined = h.t()
        else:
            neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                                  self.num_sample)
            if not self.gcn:
                if self.cuda:
                    self_feats = self.features(torch.LongTensor(nodes).cuda())
                else:
                    self_feats = self.features(torch.LongTensor(nodes))
                combined = torch.cat([self_feats, neigh_feats], dim=1)
            else:
                combined = neigh_feats

            if self.kernel == "GAT":
                combined = F.elu(self.weight.mm(combined.t()))
            else:
                combined = F.relu(self.weight.mm(combined.t()))
        return combined
