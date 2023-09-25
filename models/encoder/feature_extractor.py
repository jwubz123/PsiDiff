import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import torch_geometric.transforms as T
from torch_geometric.nn import MetaLayer
from models.utils.distributions import *
from models.dmasif_encoder.data_iteration import iterate
from models.dmasif_encoder.protein_surface_encoder import dMaSIF
from .schnet import *
from .gin import *

class NodeSampling(nn.Module):
    def __init__(self):
        super(NodeSampling, self).__init__()
          
    def forward(self, h_t, h_l=None):

        target_idx = h_t['target_idx']
        h_t['batch'] = h_t['batch'][target_idx]
        h_t['xyz'] = h_t['xyz'][target_idx]
        h_t['embedding_1'] = h_t['embedding_1'][target_idx]
        if h_l is not None:
            complex_idx = h_t['complex_idx']         
            h_l.complex_pos_batch = h_l.complex_pos_batch[complex_idx]
            h_l.complex_pos = h_l.complex_pos[complex_idx]
            return h_t, h_l  
        else:
            return h_t
            
class ResBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.15):
        super(ResBlock, self).__init__()
        
        self.projectDown_node = nn.Linear(in_channels, in_channels//4)
        self.projectDown_edge = nn.Linear(in_channels, in_channels//4)
        self.bn1_node = nn.BatchNorm1d(in_channels//4)
        self.bn1_edge = nn.BatchNorm1d(in_channels//4)
        
        self.conv = MetaLayer(edge_model=EdgeModel(in_channels//4), node_model=NodeModel(in_channels//4), global_model=None)
                
        self.projectUp_node = nn.Linear(in_channels//4, in_channels)
        self.projectUp_edge = nn.Linear(in_channels//4, in_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2_node = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_node.weight)
        self.bn2_edge = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_edge.weight)
                
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        h_node = F.elu(self.bn1_node(self.projectDown_node(x)))
        h_edge = F.elu(self.bn1_edge(self.projectDown_edge(edge_attr)))
        h_node, h_edge, _ = self.conv(h_node, edge_index, h_edge, None, batch)
        
        h_node = self.dropout(self.bn2_node(self.projectUp_node(h_node)))
        data.x = F.elu(h_node + x)
        
        h_edge = self.dropout(self.bn2_edge(self.projectUp_edge(h_edge))) 
        data.edge_attr = F.elu(h_edge + edge_attr)
        
        return data

class ResBlockComplex(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.15):
        super(ResBlockComplex, self).__init__()
        
        self.projectDown_node = nn.Linear(in_channels, in_channels//4)
        self.projectDown_edge = nn.Linear(in_channels, in_channels//4)
        self.bn1_node = nn.BatchNorm1d(in_channels//4)
        self.bn1_edge = nn.BatchNorm1d(in_channels//4)
        
        self.conv = MetaLayer(edge_model=EdgeModel(in_channels//4), node_model=NodeModel(in_channels//4), global_model=None)
                
        self.projectUp_node = nn.Linear(in_channels//4, in_channels)
        self.projectUp_edge = nn.Linear(in_channels//4, in_channels)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2_node = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_node.weight)
        self.bn2_edge = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_edge.weight)
                
    def forward(self, data):

        node_attr, edge_attr, edge_index, batch = data

        h_node = F.elu(self.bn1_node(self.projectDown_node(node_attr)))
        h_edge = F.elu(self.bn1_edge(self.projectDown_edge(edge_attr)))
        h_node, h_edge, _ = self.conv(h_node, edge_index, h_edge, None, batch)
        
        h_node = self.dropout(self.bn2_node(self.projectUp_node(h_node)))
        node_attr = F.elu(h_node + node_attr)
        
        h_edge = self.dropout(self.bn2_edge(self.projectUp_edge(h_edge))) 
        edge_attr = F.elu(h_edge + edge_attr)
        data = (node_attr, edge_attr, edge_index, batch)
        
        return data

class EdgeModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(EdgeModel, self).__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(in_channels*3, in_channels), nn.BatchNorm1d(in_channels), nn.ELU())

    def forward(self, src, dest, edge_attr, u, batch):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)

    
class NodeModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = nn.Sequential(nn.Linear(in_channels*2, in_channels), nn.BatchNorm1d(in_channels), nn.ELU())
        self.node_mlp_2 = nn.Sequential(nn.Linear(in_channels*2, in_channels), nn.BatchNorm1d(in_channels), nn.ELU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        row, col = edge_index # [E]
        out = torch.cat([x[row], edge_attr], dim=1) # [E, 2H]
        out = self.node_mlp_1(out) # [E, H]
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0)) # [N, H]
        out = torch.cat([x, out], dim=1) # [N, 2H]
        return self.node_mlp_2(out) # [N, H]

class LigandNet(nn.Module):
    def __init__(self, in_channels, edge_features=6, hidden_dim=128, residual_layers=20, dropout_rate=0.15):
        super(LigandNet, self).__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_dim)
        # self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        self.conv1 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv2 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv3 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        layers = [ResBlock(in_channels=hidden_dim, dropout_rate=dropout_rate) for i in range(residual_layers)] 
        self.resnet = nn.Sequential(*layers)
                        
    def forward(self, data, local_edge_mask):

        if local_edge_mask is not None:
            data.edge_index = data.edge_index[:, local_edge_mask]
            data.edge_attr = data.edge_attr[local_edge_mask, :]
        data.x = self.node_encoder(data.x)

        # data.edge_attr = self.edge_encoder(data.edge_attr)
        data.x, data.edge_attr, _ = self.conv1(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.x, data.edge_attr, _ = self.conv2(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.x, data.edge_attr, _ = self.conv3(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data = self.resnet(data)
                
        return data 

class SurfBindEncoder(nn.Module):
    def __init__(self, config, edge_channels):
        super(SurfBindEncoder, self).__init__()
        
        if config.encoder.ligand_net.type == 'gcn':
            self.ligand_model = LigandNet(config.encoder.ligand_net.in_channels,
                                          residual_layers=config.encoder.ligand_net.res_layers, 
                                          hidden_dim=config.encoder.ligand_net.hidden_dim, 
                                          dropout_rate=config.encoder.ligand_net.dropout_rate)
        elif config.encoder.ligand_net.type == 'gtb':

            raise NotImplementedError('Unfinished ligand encoder: %s' % config.encoder.ligand_net.type)
        elif config.encoder.ligand_net.type == 'dual':
            self.encoder_global = SchNetEncoder(
                hidden_channels=config.encoder.ligand_net.hidden_dim,
                num_filters=config.encoder.ligand_net.hidden_dim,
                num_interactions=config.encoder.ligand_net.num_convs,
                edge_channels=edge_channels,
                cutoff=config.cutoff,
                smooth=config.encoder.ligand_net.smooth_conv,
            )
            self.encoder_local = GINEncoder(
                hidden_dim=config.encoder.ligand_net.hidden_dim,
                num_convs=config.encoder.ligand_net.num_convs_local,
            )
        else:
            raise NotImplementedError('Unknown ligand encoder: %s' % config.encoder.ligand_net.type)
        if config.encoder.use_complex:
            self.target_model = dMaSIF(config.encoder.dmasif)
            if config.encoder.ligand_net.nodes_per_target > 0:
                self.node_sampling = NodeSampling()
        
    def forward(self, data_ligand, data_target, edge_length, local_edge_mask, config, device='cpu'):
        
        if config.encoder.ligand_net.type == 'gcn':
            h_l = self.ligand_model(data_ligand, local_edge_mask)
        elif config.encoder.ligand_net.type == 'dual':
            atom_type = self.onehot2label(data_ligand.x)
            h_l = data_ligand
            if local_edge_mask is None:
                node_l_attr = self.encoder_global(
                        z=atom_type,
                        edge_index=data_ligand.edge_index,
                        edge_length=edge_length,
                        edge_attr=data_ligand.edge_attr,
                    ) #(N, G)
            else:
                node_l_attr = self.encoder_local(
                    z=atom_type,
                    edge_index=data_ligand.edge_index[:, local_edge_mask],
                    edge_attr=data_ligand.edge_attr[local_edge_mask],
                ) #(N, G)
                h_l.edge_index = data_ligand.edge_index[:, local_edge_mask]
                h_l.edge_attr=data_ligand.edge_attr[local_edge_mask]
            h_l.x = node_l_attr
        else:
            raise NotImplementedError('Unknown ligand encoder: %s' % config.encoder.ligand_net.type)

        if config.encoder.use_complex:

            outputs = iterate(self.target_model,
                                data_target,
                                args=config.encoder.dmasif,
                                device=device
                            )
            h_t = outputs['P1']
            h_t['target_idx'] = data_target['target_idx']
            h_t['complex_idx'] = data_target['complex_idx']
            
            if self.node_sampling:
                if config.network == 'dualenc' or config.network == 'no_inter':
                    h_t, h_l = self.node_sampling(h_t, h_l)
                elif config.network == 'ligand_based':
                    h_t = self.node_sampling(h_t)
                else: 
                    raise NotImplementedError('Unknown network: %s' % config.network)
                
        else: 
            h_t = None
        
        return h_l, h_t
    
    def onehot2label(self, onehot_attr):
        return torch.topk(onehot_attr, 1)[1].squeeze(1) + 1


class ProteinEncoder(nn.Module):
    def __init__(self, config):
        super(ProteinEncoder, self).__init__()
        
        self.target_model = dMaSIF(config.encoder.dmasif)
        if config.encoder.ligand_net.nodes_per_target > 0:
            self.node_sampling = NodeSampling()
        
    def forward(self, data_ligand, data_target, config, device='cpu'):

        outputs = iterate(self.target_model,
                            data_target,
                            args=config.encoder.dmasif,
                            device=device
                        )
        h_t = outputs['P1']
        h_t['target_idx'] = data_target['target_idx']
        h_t['complex_idx'] = data_target['complex_idx']
        h_l = data_ligand
        if self.node_sampling:
            if config.network == 'dualenc' or config.network == 'no_inter':
                h_t, h_l = self.node_sampling(h_t, h_l)
            elif config.network == 'ligand_based':
                h_t = self.node_sampling(h_t)
            else: 
                raise NotImplementedError('Unknown network: %s' % config.network)
        return h_l, h_t
    
class ComplexEncoder(nn.Module):
    def __init__(self, hidden_dim=128, residual_layers=20, dropout_rate=0.15):
        super(ComplexEncoder, self).__init__()

        self.conv1 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.residual_layers = residual_layers
        if self.residual_layers > 0:
            self.conv2 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
            self.conv3 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
            layers = [ResBlockComplex(in_channels=hidden_dim, dropout_rate=dropout_rate) for i in range(residual_layers)] 
            self.resnet = nn.Sequential(*layers)
                        
    def forward(self, node_attr, edge_attr, edge_index, batch):

        node_attr, edge_attr, _ = self.conv1(node_attr, edge_index, edge_attr, None, batch)
        if self.residual_layers > 0:
            node_attr, edge_attr, _ = self.conv2(node_attr, edge_index, edge_attr, None, batch)
            node_attr, edge_attr, _ = self.conv3(node_attr, edge_index, edge_attr, None, batch)
            data = (node_attr, edge_attr, edge_index, batch)
            data = self.resnet(data)
            node_attr, edge_attr, edge_index, batch = data
                
        return node_attr, edge_attr 

class CatComplexLinear(nn.Module):
    def __init__(self, config):
        super(CatComplexLinear, self).__init__()

        self.linear = nn.Linear(config.encoder.complex_net.hidden_dim, config.encoder.complex_net.hidden_dim)
                        
    def forward(self, h_l, h_t):



        complex_node_attr = torch.cat([h_l.x, h_t['embedding_1']], dim=0)
        complex_node_attr = self.linear(complex_node_attr)
                
        return complex_node_attr
