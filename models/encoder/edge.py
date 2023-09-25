import torch
from torch.nn import Module, Embedding
from ..common import MultiLayerPerceptronEdge

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class GaussianSmearingEdgeEncoder(Module):

    def __init__(self, num_gaussians=64, cutoff=10.0, dict_size=100):
        super().__init__()
        #self.NUM_BOND_TYPES = 22
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.rbf = GaussianSmearing(start=0.0, stop=cutoff * 2, num_gaussians=num_gaussians)    # Larger `stop` to encode more cases
        self.bond_emb = Embedding(dict_size, embedding_dim=num_gaussians)

    @property
    def out_channels(self):
        return self.num_gaussians * 2

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        edge_attr = torch.cat([self.rbf(edge_length), self.bond_emb(edge_type)], dim=1)
        return edge_attr

class GaussianSmearingEdgeEncoderNoType(Module):

    def __init__(self, num_gaussians=64, cutoff=10.0):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.rbf = GaussianSmearing(start=0.0, stop=cutoff * 2, num_gaussians=num_gaussians)    # Larger `stop` to encode more cases

    @property
    def out_channels(self):
        return self.num_gaussians * 2

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        edge_attr = self.rbf(edge_length)
        return edge_attr

class MLPEdgeEncoder(Module):

    def __init__(self, hidden_dim=100, activation='relu', dict_size=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bond_emb = Embedding(dict_size, embedding_dim=self.hidden_dim)
        self.mlp = MultiLayerPerceptronEdge(1, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """

        d_emb = self.mlp(edge_length) # (num_edge, hidden_dim)
        edge_attr = self.bond_emb(edge_type) # (num_edge, hidden_dim)
        return d_emb * edge_attr # (num_edge, hidden)

class MLPEdgeNoTypeEncoder(Module):

    def __init__(self, hidden_dim=128, activation='relu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = MultiLayerPerceptronEdge(1, [self.hidden_dim, self.hidden_dim], activation=activation)

    @property
    def out_channels(self):
        return self.hidden_dim

    def forward(self, edge_length, edge_type):
        """
        Input:
            edge_length: The length of edges, shape=(E, 1).
            edge_type: The type pf edges, shape=(E,)
        Returns:
            edge_attr:  The representation of edges. (E, 2 * num_gaussians)
        """
        d_emb = self.mlp(edge_length) # (num_edge, hidden_dim)
        return d_emb # (num_edge, hidden)

def get_edge_encoder(cfg):
    if cfg.edge_encoder == 'mlp':
        return MLPEdgeEncoder(cfg.hidden_dim, cfg.mlp_act, dict_size=cfg.edge_order+cfg.encoder.ligand_net.edge_features+4)
    elif cfg.edge_encoder == 'gaussian':
        return GaussianSmearingEdgeEncoder(config.hidden_dim // 2, cutoff=config.cutoff, dict_size=cfg.edge_order+cfg.encoder.ligand_net.edge_features+4)
    else:
        raise NotImplementedError('Unknown edge encoder: %s' % cfg.edge_encoder)
        
