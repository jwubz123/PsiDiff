import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data, Batch
import numpy as np
from numpy import pi as PI
from tqdm.auto import tqdm
from copy import copy

from ..common import MultiLayerPerceptron, assemble_atom_pair_feature, extend_graph_order_radius, extend_graph_order_radius_complex
from ..encoder import get_edge_encoder, SurfBindEncoder, LTMPLocal, LTMPGlobal, ProteinEncoder, ComplexEncoder, CatComplexLinear
from ..geometry import get_distance, eq_transform



def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class TLPENetwork(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        edge_encoder:  Takes both edge type and edge length as input and outputs a vector
        [Note]: node embedding is done in SchNetEncoder
        """
        self.edge_encoder_global = get_edge_encoder(config)
        self.edge_encoder_local = get_edge_encoder(config)
        if config.condition_time:
            in_channels = config.encoder.ligand_net.in_channels + 1
        else:
            in_channels = config.encoder.ligand_net.in_channels

        in_channels = in_channels + len(config.condition_prop)
        self.lig_node_embed = nn.Linear(in_channels, config.encoder.ligand_net.hidden_dim)
        """
        The graph neural network that extracts node-wise features.
        """
        self.protein_encoder_global = ProteinEncoder( config=config)

        self.encoder_local = SurfBindEncoder(
            config=config,
        edge_channels = self.edge_encoder_local.out_channels
        )
            
        """
        The block that combine node-wise features of ligand and target.
        """
        if config.encoder.assembel_net_global == 'cat':
            self.feature_assembler_global = CatComplexLinear(config)
        elif config.encoder.assembel_net_global == 'ltmp':
                self.feature_assembler_global = LTMPGlobal(config.encoder.ltmp)
        else:
            raise NotImplementedError('Unknown assembeling block: %s' % config.encoder.assembel_net_global)
        if config.encoder.assembel_net_local == 'ltmp':
            self.feature_assembler_local = LTMPLocal(config.encoder.ltmp)
        elif config.encoder.assembel_net_local == 'cat':
            self.feature_assembler_local = CatComplexLinear(config)
        else:
            raise NotImplementedError('Unknown assembeling block: %s' % config.encoder.assembel_net_local)
        
        """
        The block that extract complex features.
        """
        self.complex_encoder = ComplexEncoder(residual_layers=config.encoder.complex_net.res_layers, 
                                          hidden_dim=config.encoder.complex_net.hidden_dim, 
                                          dropout_rate=config.encoder.complex_net.dropout_rate)
        """
        `output_mlp` takes a mixture of two nodewise features and edge features as input and outputs 
            gradients w.r.t. edge_length (out_dim = 1).
        """
        self.grad_global_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=config.mlp_act
        )

        self.grad_local_dist_mlp = MultiLayerPerceptron(
            2 * config.hidden_dim,
            [config.hidden_dim, config.hidden_dim // 2, 1], 
            activation=config.mlp_act
        )

        '''
        Incorporate parameters together
        '''
        self.model_global = nn.ModuleList([self.protein_encoder_global, self.lig_node_embed, self.edge_encoder_global, self.feature_assembler_global, self.complex_encoder, self.grad_global_dist_mlp])
        self.model_local = nn.ModuleList([self.edge_encoder_local, self.encoder_local, self.feature_assembler_local, self.grad_local_dist_mlp])

        self.model_type = config.type  # config.type


        # denoising diffusion
        ## betas
        betas = get_beta_schedule(
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            num_diffusion_timesteps=config.num_diffusion_timesteps,
        )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False)
        ## variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.num_timesteps = self.betas.size(0)
        if self.model_type == 'conditioned_diffusion':
            self.num_timesteps = self.num_timesteps - 1
            self.gamma, self.sigmas, self.alphas = self.cal_gamma()

    def forward(self, ligand, target, batch, time_step, cutoff, cutoff_inter,
                edge_index=None, edge_type=None, edge_length=None, return_edges=False, 
                extend_order=True, extend_radius=True):
        """
        """
        
        N = ligand.x.size(0)
        
        bond_index = ligand.edge_index
        pos = ligand.pos
        bond_type = onehot2label(ligand.edge_attr)
        
        ligand = self.add_cond(ligand, time_step, device=pos.device)
        
        ligand_temp_global = copy(ligand)
        ligand_temp_local = copy(ligand)
        ### extend ligand graph
        if edge_index is None or edge_type is None or edge_length is None:
            edge_index, edge_type = extend_graph_order_radius(
                num_nodes=N,
                pos=pos,
                ptr=ligand.ptr,
                edge_index=bond_index,
                edge_type=bond_type,
                batch=batch,
                num_types=self.config.encoder.ligand_net.edge_features,
                order=self.config.edge_order,
                cutoff=cutoff,
                extend_order=extend_order,
                extend_radius=extend_radius
            ) # edge_type = bond_type + order, if radius, edge_index added, edge_type = 0
            edge_length = get_distance(pos, edge_index).unsqueeze(-1)   # (E, 1)
            ligand_temp_global.edge_index = edge_index
            ligand_temp_local.edge_index = edge_index
        local_edge_mask = is_local_edge(edge_type)  # (E, ) # only edges with bonds in order
        ### Protein graph encoding and downsampling
        h_l_global, h_t_global = self.protein_encoder_global(ligand_temp_global, target, self.config,
            device=pos.device)
        ### Embed ligand node
        ligand_temp_global.x = self.lig_node_embed(ligand_temp_global.x)
        target.complex_index_batch = h_l_global.complex_pos_batch
        # Local
        edge_attr_local = self.edge_encoder_local(
            edge_length=edge_length,
            edge_type=edge_type
        )   # Embed edges (E, G)
        ligand_temp_local.edge_attr = edge_attr_local

        # Node encoder Local
        h_l_local, h_t_local = self.encoder_local(
            ligand_temp_local, 
            target,
            edge_length=edge_length,
            config=self.config,
            local_edge_mask=local_edge_mask,
            device=pos.device
            ) #(N, G)
        node_l_attr_local = h_l_local.x
        edge_attr_local = h_l_local.edge_attr

        ### construct complex graph
        complex_edge_index, complex_local_edge_mask, inter_edge_mask, complex_edge_type = extend_graph_order_radius_complex(
            lig=h_l_global, 
            tar=h_t_global, 
            pos=h_l_global.complex_pos,
            edge_index=edge_index,
            edge_type=edge_type,
            batch=h_l_global.complex_pos_batch,
            cutoff=cutoff_inter
        )
        complex_edge_length = get_distance(h_l_global.complex_pos, complex_edge_index).unsqueeze(-1)
        
       
        # Encoding global edges
        complex_edge_attr_global = self.edge_encoder_global(
            edge_length=complex_edge_length,
            edge_type=complex_edge_type
        )   # Embed edges (E, G)

        ### complex node assembler
        node_complex_attr_global = self.feature_assembler_global(h_l_global, h_t_global) # torch.Size([B, N_t, N_l, H])
        
        ### Complex encoder
        node_complex_attr_global, complex_edge_attr_global = self.complex_encoder(
            node_complex_attr_global, 
            complex_edge_attr_global,
            complex_edge_index,
            h_l_global.complex_pos_batch)

        # Assemble ligand and target features global

        h_complex_pair_global = assemble_atom_pair_feature(
            node_attr=node_complex_attr_global,
            edge_index=complex_edge_index,
            edge_attr=complex_edge_attr_global,
        )    # (E_global, 2H) #(E, 2G)
        edge_complex_inv_global = self.grad_global_dist_mlp(h_complex_pair_global)
        
        # Assemble ligand and target node features local
        node_complex_attr_local = self.feature_assembler_local(h_l_local, h_t_local)
    
        ## Assemble pairwise features
        
        h_complex_pair_local = assemble_atom_pair_feature(
            node_attr=node_complex_attr_local,
            edge_index=complex_edge_index[:, complex_local_edge_mask],
            edge_attr=edge_attr_local,
        )    # (E_global, 2H) #(E, 2G)
                
        edge_complex_inv_local = self.grad_local_dist_mlp(h_complex_pair_local)

        if return_edges:
            return edge_complex_inv_global, edge_complex_inv_local, complex_edge_index, complex_edge_length, complex_local_edge_mask
        else:
            return edge_complex_inv_global, edge_complex_inv_local
    
    def add_cond(self, ligand, time_step, device):
        ### Add time step to ligand node features
        if self.config.condition_time:
            # t is different over the batch dimension.

            h_time = time_step.index_select(0, ligand.batch).unsqueeze(-1)  # (N, 1)
            ligand.x = torch.cat([ligand.x, h_time], dim=1)

        for prop_name_str in self.config.condition_prop:
            if prop_name_str == 'prop_gap':
                context = ligand.prop_gap.float().to(device=device)
                context = context.index_select(0, ligand.batch).unsqueeze(-1)
                mean_context = self.config.energy_model.norm_gap[0]
                std_context = self.config.energy_model.norm_gap[1]
            elif prop_name_str == 'prop_energy':
                context = ligand.prop_energy.to(device=device)
                context = context.index_select(0, ligand.batch).unsqueeze(-1)
                mean_context = self.config.energy_model.norm_energy[0]
                std_context = self.config.energy_model.norm_energy[1]
            elif prop_name_str == 'atom_charges':
                context = ligand.atom_charges.to(device=device)
                context = context.unsqueeze(-1)
                mean_context = self.config.energy_model.norm_charge[0]
                std_context = self.config.energy_model.norm_charge[1]
            else: 
                raise NotImplementedError(prop_name_str)
            context = (context - mean_context) / std_context
            ligand.x = torch.cat([ligand.x, context], dim=1)
        return ligand
    
    def get_loss(self, ligand, target, num_graphs,
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, change_loss_weight=False):
        batch = ligand.batch
        bond_index = ligand.edge_index
        ligand.x = ligand.x / self.config.node_type_norm

        norm_method = 'std'
        cutoff = self.config.cutoff / target.std
        cutoff_inter = self.config.cutoff_inter / target.std
        pos = ligand.pos
        bond_type = onehot2label(ligand.edge_attr)
        complex_pos = ligand.complex_pos
       
        return self.get_loss_diffusion(ligand, target, pos, complex_pos, batch, num_graphs, cutoff, cutoff_inter,
            anneal_power, return_unreduced_loss, return_unreduced_edge_loss, extend_order, extend_radius, change_loss_weight)
        
    def get_loss_diffusion(self, ligand, target, pos, complex_pos, batch, num_graphs, cutoff, cutoff_inter,
                 anneal_power=2.0, return_unreduced_loss=False, return_unreduced_edge_loss=False, extend_order=True, extend_radius=True, change_loss_weight=False):

        self.batch = batch
        self.bs = len(ligand.ptr) - 1

        pos_input = pos.clone()
        if self.model_type == 'diffusion':
            # Four elements for DDPM: original_data(pos), gaussian_noise(pos_noise), beta(sigma), time_step
            # Sample noise levels
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs//2+1, ), device=pos.device) 
            time_step = torch.cat(
                [time_step, self.num_timesteps-time_step-1], dim=0)[:num_graphs] 
            a = self.alphas.index_select(0, time_step)  # (G, ) 
            
            # Perterb pos
            a_pos = a.index_select(0, batch).unsqueeze(-1)  # (N, 1)  
            pos_noise = torch.zeros(size=pos.size(), device=pos.device) 
            pos_noise.normal_()
            pos_perturbed = a_pos.sqrt() * pos + pos_noise * (1.0 - a_pos).sqrt()
        
        elif self.model_type == 'conditioned_diffusion':
            pos_perturbed, pos_noise, a, gamma_t, time_step = self.cal_perturbed_pos(pos, num_graphs)
        
        complex_pos_perturbed = torch.zeros_like(complex_pos)
        counts = torch.unique(ligand.batch, return_counts=True)[1]

        for i in range(len(ligand.ptr)-1):
            complex_pos_perturbed[ligand.complex_pos_ptr[i]:(ligand.complex_pos_ptr[i]+counts[i])] = pos_perturbed[ligand.ptr[i]:ligand.ptr[i+1]]
            complex_pos_perturbed[(ligand.complex_pos_ptr[i]+counts[i]):ligand.complex_pos_ptr[i+1]] = complex_pos[(ligand.complex_pos_ptr[i]+counts[i]):ligand.complex_pos_ptr[i+1]]

        # update ligand pos
        ligand.pos = pos_perturbed
        ligand.complex_pos = complex_pos_perturbed
        
        # Normalize time_step:
        time_step = time_step / self.num_timesteps
        
        # Update invariant edge features
        edge_complex_inv_global, edge_complex_inv_local, complex_edge_index, complex_edge_length, complex_local_edge_mask = self(
            ligand, 
            target,
            batch, 
            time_step,
            cutoff,
            cutoff_inter,
            return_edges = True,
            extend_order = extend_order,
            extend_radius = extend_radius
        )   # (E_global, 1), (E_local, 1)
        
        complex_pos_perturbed = complex_pos_perturbed[target['complex_idx']]

        max_prev_batch_complex = 0
        complex_counts = torch.unique(target.complex_index_batch, return_counts=True)[1]
        ligand_mask = torch.zeros(size=target['complex_idx'].size(), device=pos.device)
        for i in range(len(ligand.ptr)-1):
            ligand_mask[max_prev_batch_complex:max_prev_batch_complex+counts[i]]=1
            max_prev_batch_complex += complex_counts[i]
        ligand_mask = (ligand_mask == 1)
        
        loss, loss_global, loss_local = self.cal_edge_to_pos_loss(a, target.complex_index_batch, complex_edge_index, complex_pos, complex_pos_perturbed, complex_edge_length, edge_complex_inv_global, edge_complex_inv_local, cutoff, target['complex_idx'], complex_local_edge_mask, ligand_mask, change_loss_weight) 

        
        if self.model_type == 'conditioned_diffusion':
            kl_prior_loss = self.kl_prior(pos_input, ligand.batch)

            loss = loss / 800 + kl_prior_loss
        
        if return_unreduced_edge_loss:
            pass
        elif return_unreduced_loss:
            return loss, loss_global, loss_local
        else:
            return loss
    
    def cal_edge_to_pos_loss(self, a, batch, edge_index, pos, pos_perturbed, edge_length, edge_inv_global, edge_inv_local, cutoff, complex_idx, local_edge_mask, ligand_mask, change_loss_weight):
        
        edge2graph = batch.index_select(0, edge_index[0]) # torch.Size([E]), values repeat batch indices 0 to 127
        # Compute sigmas_edge
        a_edge = a.index_select(0, edge2graph)  # (E, 1)
        if len(a_edge.shape) == 1:
            a_edge = a_edge.unsqueeze(-1)

        # Compute original and perturbed distances
        d_gt =  get_distance(pos[complex_idx], edge_index).unsqueeze(-1)   # (E, 1)
        d_perturbed = edge_length

        if self.config.edge_encoder == 'gaussian':
            # Distances must be greater than 0 
            d_sgn = torch.sign(d_perturbed)
            d_perturbed = torch.clamp(d_perturbed * d_sgn, min=0.01, max=float('inf'))

        d_target = (d_perturbed - d_gt * a_edge.sqrt()) / (1.0 - a_edge).sqrt()


        global_mask = ~local_edge_mask.unsqueeze(-1)
        target_d_global = torch.where(global_mask, d_target, torch.zeros_like(d_target))
        edge_inv_global = torch.where(global_mask, edge_inv_global, torch.zeros_like(edge_inv_global))
        target_pos_global = eq_transform(target_d_global, pos_perturbed, edge_index, edge_length, self.config.eq_transform_type)
        node_eq_global = eq_transform(edge_inv_global, pos_perturbed, edge_index, edge_length, self.config.eq_transform_type)

        loss_global = (node_eq_global - target_pos_global)**2
        loss_global = loss_global[ligand_mask]
        loss_global = torch.sum(loss_global, dim=-1, keepdim=True)

        target_pos_local = eq_transform(d_target[local_edge_mask], pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask], self.config.eq_transform_type)
        node_eq_local = eq_transform(edge_inv_local, pos_perturbed, edge_index[:, local_edge_mask], edge_length[local_edge_mask], self.config.eq_transform_type)

        loss_local = (node_eq_local - target_pos_local)**2
        loss_local = loss_local[ligand_mask]
        loss_local = torch.sum(loss_local, dim=-1, keepdim=True)

        if change_loss_weight:
            loss = loss_global
        else:
            loss = self.config.w_global * loss_global + loss_local
        return loss, loss_global, loss_local
    


    def cal_perturbed_pos(self, pos, num_graphs):
        # Sample a timestep t.
        t_int = torch.randint(
            0, self.num_timesteps + 1, size=(num_graphs, ), device=pos.device).float()
        s_int = t_int - 1
        self.t_is_zero = (t_int == 0).float()  # Important to compute log p(x | z0).
        self.gamma, self.sigmas, self.alphas = self.gamma.to(pos.device), self.sigmas.to(pos.device), self.alphas.to(pos.device)

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s = s_int / self.num_timesteps
        t = t_int / self.num_timesteps

        # Compute gamma_s and gamma_t via the network.
        
        gamma_s = self.inflate_batch_array(self.get_gamma_t(s)) #torch.Size([N, 1])
        gamma_t = self.inflate_batch_array(self.get_gamma_t(t))

        # Compute alpha_t and sigma_t from gamma.
        alpha_t = self.cal_alpha(gamma_t) #torch.Size([N, 1])
        sigma_t = self.cal_sigma(gamma_t)

        pos_noise = torch.zeros(size=pos.size(), device=pos.device) #[2436, 3]
        pos_noise.normal_()
        
        # Change alpha_t and sigma_t to sparse

        # Sample pos_perturbed given pos for timestep t, from q(pos_perturbed | pos)
        pos_perturbed = alpha_t * pos + sigma_t * pos_noise
        return pos_perturbed, pos_noise, alpha_t, gamma_t, t
    
    def kl_prior(self, pos, batch):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).
        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # pos, node_mask = to_dense_batch(pos, batch, fill_value=0)
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((self.bs), device=pos.device)
        gamma_T = self.get_gamma_t(ones)
        gamma_T = self.inflate_batch_array(gamma_T)
        alpha_T = self.cal_alpha(gamma_T)
        
        # Compute means.
        mu_T = alpha_T * pos

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_x = self.cal_sigma(gamma_T).squeeze()  # Remove inflate, only keep batch dimension for x-part.
        

        # Compute KL for x-part.
        zeros, ones = torch.zeros_like(mu_T), torch.ones_like(sigma_T_x)
        subspace_d = self.subspace_dimensionality(pos.size(0))
        kl_distance_x = gaussian_KL_for_dimension(mu_T, sigma_T_x, zeros, ones, d=subspace_d)

        return kl_distance_x.unsqueeze(-1)

    def cal_gamma(self):
        alphas2 = self.alphas
        print('alphas2', alphas2)
        # alphas2 [0.99999496 0.9999899  0.99998483 ... 0.00673504 0.0067216  0.00670819]
        sigmas2 = 1 - alphas2
        # sigmas [5.04499905e-06, 1.01018278e-05, 1.51705143e-05, ...,9.93264963e-01, 9.93278400e-01, 9.93291810e-01]
        log_alphas2 = torch.log(alphas2)
        log_sigmas2 = torch.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # gamma [-12.19710805, -11.50278408, -11.09614169, ...,   4.99367414, 4.99568471,   4.99769526]
        print('gamma', -log_alphas2_to_sigmas2)
        # sigmas2 = torch.from_numpy(sigmas2).float()
        sigmas2 = torch.nn.Parameter(sigmas2, requires_grad=False)
        # alphas2 = torch.from_numpy(alphas2).float()
        alphas2 = torch.nn.Parameter(alphas2, requires_grad=False)
        gamma = torch.nn.Parameter(-log_alphas2_to_sigmas2,
            requires_grad=False)
        return gamma, sigmas2, alphas2

    def inflate_batch_array(self, array):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        array = array.index_select(0, self.batch)
        if len(array.shape) == 1:
            array = array.unsqueeze(-1)
        # target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1).view(target_shape)
        return array

    def cal_sigma(self, gamma):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)))
    
    def get_gamma_t(self, t):
        t_int = torch.round(t * self.num_timesteps).long()
        return self.gamma[t_int]
    
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor, device='cpu'):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t))
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = nn.functional.logsigmoid(-gamma_t)
        log_alpha2_s = nn.functional.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s.to(device), sigma_t_given_s.to(device), alpha_t_given_s.to(device)

    def cal_alpha(self, gamma):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)))

    def subspace_dimensionality(self, number_of_nodes):
        """Compute the dimensionality on translation-invariant linear subspace where distributions on x are defined."""
        return (number_of_nodes - 1) * 3

def is_local_edge(edge_type):
    return edge_type > 0

def onehot2label(onehot_attr):
    return torch.topk(onehot_attr, 1)[1].squeeze(1) + 1

def center_pos(pos, batch):
    pos_center = pos - scatter_mean(pos, batch, dim=0)[batch]
    return pos_center

# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)

def gaussian_KL_for_dimension(q_mu, q_sigma, p_mu, p_sigma, d):
    """Computes the KL distance between two normal distributions.

        Args:
            q_mu: Mean of distribution q.
            q_sigma: Standard deviation of distribution q.
            p_mu: Mean of distribution p.
            p_sigma: Standard deviation of distribution p.
        Returns:
            The KL distance, summed over all dimensions except the batch dim.
        """
    mu_norm2 = ((q_mu - p_mu)**2).sum(-1)
    assert len(q_sigma.size()) == 1
    assert len(p_sigma.size()) == 1
    return d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d