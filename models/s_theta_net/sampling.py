import torch
import numpy as np
from tqdm.auto import tqdm
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from torch import nn
import torch.autograd as autograd

from .s_theta import TLPENetwork, get_beta_schedule
from ..geometry import eq_transform
from models.g_phi_net.utils import RequiresGradContext
from utils.energy_utils import remove_mean_with_mask


class SamplingWithGuidance(torch.nn.Module):

    def __init__(self, 
                config, 
                gen_model, 
                guidance_model1,
                guidance_model2,
                guidance_model3,
                cond_type,
                n_steps,
                step_lr,
                w_global,
                global_start_sigma,
                clip,
                clip_local,
                extend_order=3, 
                extend_radius=True):
        super().__init__()
        self.config = config
        self.gen_model = gen_model
        self.guidance_model1 = guidance_model1
        self.guidance_model2 = guidance_model2
        self.guidance_model3 = guidance_model3
        self.cond_type = cond_type
        self.n_steps= n_steps
        self.step_lr=step_lr
        self.w_global=w_global
        self.global_start_sigma=global_start_sigma
        self.clip=clip
        self.clip_local=clip_local
        self.extend_order = extend_order
        self.extend_radius=extend_radius 
        self.n_dims = 3
    
    def sampling(self, ligand, target):
        self.batch = ligand.batch
        self.cutoff = self.config.cutoff / target.std
        self.cutoff_inter = self.config.cutoff_inter / target.std
        self.bs = len(ligand.ptr) - 1
        self.device = device=ligand.pos.device

        sqrt_var = self.cal_beta_alpha()
        self.gamma = self.cal_gamma(device=ligand.pos.device)
        ''' torch.Size([energy_model.diffusion_steps + 1]), torch.Size([5000])
        self.gamma: tensor([-12.1971, -11.5028, -11.0961,  ...,   4.9937,   4.9957,   4.9977])
        self.sigmas: tensor([5.0450e-06, 1.0102e-05, 1.5171e-05,  ..., 9.9326e-01, 9.9328e-01, 9.9329e-01])
        self.alphas: tensor([1.0000, 1.0000, 1.0000,  ..., 0.0067, 0.0067, 0.0067])
        '''
        
        ### random initialize coordinates (normalized) x_0
        self.N = ligand.pos.size(0)
        pos_init, z, node_mask, edge_mask, out_mask = self.init_z(self.N, ligand)
        ### cal # of atoms and context
        label1, label2, label3 = self.cal_label(ligand)
        context1, context2, context3 = self.cal_cond(ligand)
        ligand = self.init_complex_pos(ligand, pos_init)
        
        ### sample with guidance p(xs|xt) ddpm
        ### cal eps_t(x) from eps network (generation network)
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        ligand_mask_down = None
        
        # for s in reversed(range(1, self.n_steps)):
        
        for s in tqdm(reversed(range(0, self.n_steps)), desc = 'sampling', total = self.n_steps):
            ### s start from self.n_steps-1, t from self.n_steps should be 4999 here
            s_array = torch.full((self.bs, 1), fill_value=s, device=z.device)
            t_array = s_array + 1
            s_array = s_array / self.n_steps
            t_array = t_array / self.n_steps
            z, ligand, ligand_mask_down = self.sample_p_zs_given_zt(sqrt_var, s, s_array, t_array, z, node_mask, edge_mask, out_mask, context1, context2, context3, label1, label2, label3, ligand, target, ligand_mask_down, fix_noise=False)
            
        ### sample p(x | x_0)​
        # Finally sample p(x, h | z_0).
        pos = self.sample_p_xh_given_z0(z, node_mask, edge_mask, ligand, target, ligand_mask_down, fix_noise=False)
        # pos = z[:,:,:self.n_dims]
        ### unnormalized back transition of position
        pos = pos[node_mask.squeeze(-1)]
        pos_tr_back = inverse_normalize(pos, target.mu, target.std, self.bs, ligand.batch)
        return pos_tr_back

    def cal_beta_alpha(self):
        betas = get_beta_schedule(
                beta_schedule=self.config.beta_schedule,
                beta_start=self.config.beta_start,
                beta_end=self.config.beta_end,
                num_diffusion_timesteps=self.config.num_diffusion_timesteps,
            )
        betas = torch.from_numpy(betas).float()
        self.betas = nn.Parameter(betas, requires_grad=False).to(self.device)
        ## variances
        alphas = (1. - betas).cumprod(dim=0)
        self.alphas = nn.Parameter(alphas, requires_grad=False).to(self.device)
        alphas_cumprod = self.alphas
        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sigmas = (1.0 - self.alphas).sqrt() / self.alphas.sqrt()
        sqrt_var = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return sqrt_var.to(self.device)
    
    def cal_gamma(self, device='cpu'):
        
        sigmas2 = 1 - self.alphas
        # sigmas [5.04499905e-06, 1.01018278e-05, 1.51705143e-05, ...,9.93264963e-01, 9.93278400e-01, 9.93291810e-01]
        log_alphas2 = torch.log(self.alphas)
        log_sigmas2 = torch.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2

        # gamma [-12.19710805, -11.50278408, -11.09614169, ...,   4.99367414, 4.99568471,   4.99769526]
        # print('gamma', -log_alphas2_to_sigmas2)
        
        gamma = torch.nn.Parameter(-log_alphas2_to_sigmas2,
            requires_grad=False)
        return gamma.to(self.device)
    
    def get_gamma_t(self, t):
        t_int = torch.round(t * self.n_steps).long()
        return self.gamma[t_int]
    
    def init_z(self, N, ligand):
        '''
        initialize z with shape (B, max_num_nodes_in_batch, num_atom_type + 3)
        '''

        z_x, pos_init, pos_mask = sample_center_gravity_zero_gaussian_with_mask(N,
            device=ligand.pos.device, batch=ligand.batch)
        norm_biases = self.config.energy_model.norm_biases
        norm_values = self.config.energy_model.normalization_factor[1]
        z_h, node_mask = get_zh_init_with_mask(ligand.x, ligand.pos.device, ligand.batch, norm_biases, norm_values)
        edge_mask, out_mask = self.cal_masks(node_mask, z_x)
        z = torch.cat([z_x, z_h], dim=2)

        return pos_init, z, node_mask, edge_mask, out_mask

    def init_complex_pos(self, ligand, pos_init):
        pos = pos_init
        complex_pos = ligand.complex_pos
        counts = torch.unique(ligand.batch, return_counts=True)[1]
        for i in range(len(ligand.ptr)-1):
            complex_pos[ligand.complex_pos_ptr[i]:(ligand.complex_pos_ptr[i]+counts[i])] = pos[ligand.ptr[i]:ligand.ptr[i+1]]
        ligand.pos = pos
        ligand.complex_pos = complex_pos
        return ligand

    def cal_coord_eps(self, ligand, target, time_step, current_sigma, ligand_mask_down=None):

        edge_complex_inv_global, edge_complex_inv_local, complex_edge_index, complex_edge_length, complex_local_edge_mask = self.gen_model(
            ligand, 
            target, 
            self.batch, 
            time_step=time_step,
            cutoff=self.cutoff,
            cutoff_inter=self.cutoff_inter,
            return_edges=True,
            extend_order=self.extend_order,
            extend_radius=self.extend_radius
        )   # (E_global, 1), (E_local, 1)
        
        # Local
        complex_pos_down = ligand.complex_pos[target['complex_idx']]
        node_complex_eq_local = eq_transform(edge_complex_inv_local, complex_pos_down, complex_edge_index[:, complex_local_edge_mask], complex_edge_length[complex_local_edge_mask])
        if self.clip_local is not None:
            node_complex_eq_local = clip_norm(node_complex_eq_local, limit=self.clip_local)
        # Global
        if current_sigma < self.global_start_sigma:
            edge_complex_inv_global = edge_complex_inv_global * (1-complex_local_edge_mask.view(-1, 1).float())
            node_complex_eq_global = eq_transform(edge_complex_inv_global, complex_pos_down, complex_edge_index, complex_edge_length)
            node_complex_eq_global = clip_norm(node_complex_eq_global, limit=self.clip)
        else:
            node_complex_eq_global = 0
        # Sum
        eps_pos_complex = node_complex_eq_local + node_complex_eq_global * self.w_global # + eps_pos_reg * w_reg
        
        ### cal ligand_mask_down for the first time step
        if ligand_mask_down is None:
            counts = torch.unique(ligand.batch, return_counts=True)[1]
            complex_counts = torch.unique(target.complex_index_batch, return_counts=True)[1]
            max_prev_batch_complex = 0
            ligand_mask = torch.zeros(size=ligand.complex_pos_batch.size(), device=ligand.pos.device)
            ligand_mask_down = torch.zeros(size=target['complex_idx'].size(), device=ligand.pos.device)
            for idx in range(len(ligand.ptr)-1):
                ligand_mask[ligand.complex_pos_ptr[idx]:ligand.complex_pos_ptr[idx]+counts[idx]]=1
                ligand_mask_down[max_prev_batch_complex:max_prev_batch_complex+counts[idx]]=1
                max_prev_batch_complex += complex_counts[idx]
            ligand_mask = (ligand_mask == 1) 
            protein_mask = (ligand_mask == 0)
            ligand_mask_down = (ligand_mask_down == 1)      
            self.ligand_mask = ligand_mask              
        
        eps_pos = eps_pos_complex[ligand_mask_down]
        
        return eps_pos, ligand_mask_down

    def update_ligand(self, ligand, pos, ligand_mask, node_mask):
        pos = pos[node_mask.squeeze(-1)]
        ligand.pos = pos       
        ligand.complex_pos[ligand_mask] = pos

        ligand.pos = pos

        return ligand

    def inflate_batch_array(self, array, target):
        """
        Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
        axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
        """
        target_shape = (array.size(0),) + (1,) * (len(target.size()) - 1)
        return array.view(target_shape)

    
    def sigma_and_alpha_t_given_s(self, gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_tensor: torch.Tensor, device='cpu'):
        """
        Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.
        These are defined as:
            alpha t given s = alpha t / alpha s,
            sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
        """
        sigma2_t_given_s = self.inflate_batch_array(
            -expm1(softplus(gamma_s) - softplus(gamma_t)), target_tensor
        )

        # alpha_t_given_s = alpha_t / alpha_s
        log_alpha2_t = F.logsigmoid(-gamma_t)
        log_alpha2_s = F.logsigmoid(-gamma_s)
        log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

        alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
        alpha_t_given_s = self.inflate_batch_array(
            alpha_t_given_s, target_tensor)

        sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

        return sigma2_t_given_s.to(device), sigma_t_given_s.to(device), alpha_t_given_s.to(device)

    def compute_x_pred(self, net_out, zt, gamma_t):
        """Commputes x_pred, i.e. the most likely prediction of x."""

        sigma_t = self.sigma(gamma_t, target_tensor=net_out)
        alpha_t = self.alpha(gamma_t, target_tensor=net_out)
        eps_t = net_out
        x_pred = 1. / alpha_t * (zt[:,:,:self.n_dims] - sigma_t * eps_t)


        return x_pred
    
    def sigma(self, gamma, target_tensor):
        """Computes sigma given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_tensor)

    def alpha(self, gamma, target_tensor):
        """Computes alpha given gamma."""
        return self.inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_tensor)
    
    def SNR(self, gamma):
        """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
        return torch.exp(-gamma)
    
    def sample_p_xh_given_z0(self, z0, node_mask, edge_mask, ligand, target, ligand_mask_down, fix_noise=False):
        """Samples x ~ p(x|z0)."""
        zeros = torch.zeros(size=(z0.size(0), 1), device=z0.device)
        
        gamma_0 = self.get_gamma_t(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma_x = self.SNR(-0.5 * gamma_0).unsqueeze(1)
        # net_out = self.phi(z0, zeros, node_mask, edge_mask, context)

        with torch.no_grad():
            net_out, ligand_mask_down = self.cal_coord_eps(ligand, target, zeros.squeeze(-1), self.sigmas[0], ligand_mask_down=ligand_mask_down)
        # Compute mu for p(zs | zt).
        net_out, _ = to_dense_batch(net_out, ligand.batch, fill_value=0)
        
        mu_x = self.compute_x_pred(net_out, z0, gamma_0)
        x = self.sample_normal(mu=mu_x, sigma=sigma_x, node_mask=node_mask, fix_noise=fix_noise)

        return x

    def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
        """Samples from a Normal distribution."""
        bs = 1 if fix_noise else mu.size(0)
        eps, pos_init, pos_mask = sample_center_gravity_zero_gaussian_with_mask(self.N, device=node_mask.device, batch=self.batch)
        return mu + sigma * eps

    def sample_p_zs_given_zt(self, sqrt_var, time_step, s, t, zt, node_mask, edge_mask, out_mask, context1, context2, context3, label1, label2, label3, ligand, target, ligand_mask_down, fix_noise=False):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        
        gamma_s = self.get_gamma_t(s)
        gamma_t = self.get_gamma_t(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = \
            self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, zt, device=zt.device)

        with torch.no_grad():
            eps_t, ligand_mask_down = self.cal_coord_eps(ligand, target, s.squeeze(-1), self.sigmas[time_step], ligand_mask_down=ligand_mask_down)

        eps_t, _ = to_dense_batch(eps_t, ligand.batch, fill_value=0)
        pos_t = zt[:,:,:self.n_dims]
        
        pos_noise = torch.randn_like(pos_t) # (N, 3)
        ### mean for pos
        mu = torch.sqrt(1.0 / (1 - self.betas[time_step])) * (pos_t - self.betas[time_step] * eps_t / torch.sqrt(1. - self.alphas[time_step]))
        
        if self.config.energy_model.l1 > 0:
            #guidance
            weight_t = sigma2_t_given_s / alpha_t_given_s

            with RequiresGradContext(zt, requires_grad=True):
                prediction1 = self.guidance_model1.phi(zt, t, node_mask, edge_mask, context1) # torch.Size(B)
                if len(label1.shape) > 1:
                    prediction1 = prediction1[node_mask.squeeze(-1)]
                    label1 = label1[node_mask.squeeze(-1)]

                energy1 = l2_square(prediction1, label1)
                grad1 = autograd.grad(energy1.sum(), zt)[0]
            
            grad1 = remove_mean_with_mask(grad1[:, :, :self.n_dims], node_mask)


            mu = mu - self.config.energy_model.l1 * weight_t * grad1.detach()
        else:
            if time_step == (self.n_steps - 1):
                print('Not using guidance1!')
        
        if self.config.energy_model.l2 > 0 and self.guidance_model2 is not None:
            #guidance
            weight_t = sigma2_t_given_s / alpha_t_given_s

            with RequiresGradContext(zt, requires_grad=True):
                prediction2 = self.guidance_model2.phi(zt, t, node_mask, edge_mask, context2) # torch.Size(B)
                if len(label2.shape) > 1:
                    prediction2 = prediction2[node_mask.squeeze(-1)]
                    label2 = label2[node_mask.squeeze(-1)]
                energy2 = l2_square(prediction2, label2)
                grad2 = autograd.grad(energy2.sum(), zt)[0]
            
            grad2 = remove_mean_with_mask(grad2[:, :, :self.n_dims], node_mask)
            mu = mu - self.config.energy_model.l2 * weight_t * grad2.detach()
        else:
            if time_step == (self.n_steps - 1):
                print('Not using guidance2!')

        if self.config.energy_model.l3 > 0 and self.guidance_model3 is not None:
            #guidance
            weight_t = sigma2_t_given_s / alpha_t_given_s

            with RequiresGradContext(zt, requires_grad=True):
                prediction3 = self.guidance_model3.phi(zt, t, node_mask, edge_mask, context3) # torch.Size(B)
                if len(label3.shape) > 1:
                    prediction3 = prediction3[node_mask.squeeze(-1)]
                    label3 = label3[node_mask.squeeze(-1)]
                
                energy3 = l2_square(prediction3, label3)
                grad3 = autograd.grad(energy3.sum(), zt)[0]
            grad3 = remove_mean_with_mask(grad3[:, :, :self.n_dims], node_mask)
            mu = mu - self.config.energy_model.l3 * weight_t * grad3.detach()
        else:
            if time_step == (self.n_steps - 1):
                print('Not using guidance3!')
        
        ### update
        zs = mu + sqrt_var[time_step] * pos_noise

        # Project down to avoid numerical runaway of the center of gravity.
        pos = remove_mean_with_mask(zs, node_mask)
        zs = torch.cat(
                [pos, zt[:, :, self.n_dims:]], dim=2
            )
        
        ligand = self.update_ligand(ligand, pos, self.ligand_mask, node_mask)
        return zs, ligand, ligand_mask_down

    def cal_cond(self, ligand):
        context1 = context2 = context3 = None
        context_list = []
        bs, n_nodes, _ = to_dense_batch(ligand.x, ligand.batch)[0].shape
        for prop_name_str in self.config.energy_model.condition_prop:

            if prop_name_str == 'prop_gap':
                prop = ligand.prop_gap.float().to(self.device)
                prop = prop.view(bs, 1).repeat(1, n_nodes)
                prop = prop.view(bs * n_nodes, 1)
                mean_prop = self.config.energy_model.norm_gap[0]
                std_prop = self.config.energy_model.norm_gap[1]
            elif prop_name_str == 'prop_energy':
                prop = ligand.prop_energy.to(self.device)
                prop = prop.view(bs, 1).repeat(1, n_nodes)
                prop = prop.view(bs * n_nodes, 1)
                mean_prop = self.config.energy_model.norm_energy[0]
                std_prop = self.config.energy_model.norm_energy[1]
            elif prop_name_str == 'atom_charges':
                prop = ligand.atom_charges
                prop, _ = to_dense_batch(prop, ligand.batch)
                bs, n_prop = prop.shape
                if prop.shape[0] != n_nodes:
                    prop_temp = torch.zeros(bs, n_nodes)
                    prop_temp[:, :n_prop] = prop
                    prop = prop_temp
                prop = prop.view(bs * n_nodes, 1).to(self.device)
                mean_prop = self.config.energy_model.norm_charge[0]
                std_prop = self.config.energy_model.norm_charge[1]
            else: 
                raise NotImplementedError(prop_name_str)
            prop = (prop - mean_prop) / std_prop
            context_list.append(prop)
        if self.cond_type == 'multi':
            context1 = context2 = context3 = torch.cat(context_list, dim=1)
        elif self.cond_type == 'each':
            if len(self.config.energy_model.guidance_prop) >= 3:
                context3 = context_list[2]
            if len(self.config.energy_model.guidance_prop) >= 2:
                context2 = context_list[1]
            if len(self.config.energy_model.guidance_prop) >= 1:
                context1 = context_list[0]
        return context1, context2, context3

    def cal_label(self, ligand):
        label1 = None
        label2 = None
        label3 = None
        label_list = []
        for prop_name_str in self.config.energy_model.guidance_prop:
            if prop_name_str == 'prop_gap':
                properties = ligand.prop_gap.float().to(self.device)
                mean_label = self.config.energy_model.norm_gap[0]
                std_label = self.config.energy_model.norm_gap[1]
            elif prop_name_str == 'prop_energy':
                properties = ligand.prop_energy.to(self.device)
                mean_label = self.config.energy_model.norm_energy[0]
                std_label = self.config.energy_model.norm_energy[1]
            elif prop_name_str == 'atom_charges':
                properties = ligand.atom_charges.to(self.device)
                properties, _ = to_dense_batch(properties, ligand.batch.to(self.device))
                batch_size, n_prop = properties.shape
                mean_label = self.config.energy_model.norm_charge[0]
                std_label = self.config.energy_model.norm_charge[1]
            else: 
                raise NotImplementedError(prop_name_str)
            label_list.append(((properties - mean_label) / std_label).to(self.device))
        if len(self.config.energy_model.guidance_prop) >= 1:
            label1 = label_list[0]
        if len(self.config.energy_model.guidance_prop) >= 2:
            label2 = label_list[1]
        if len(self.config.energy_model.guidance_prop) >= 3:
            label3 = label_list[2]
        return label1, label2, label3

    def cal_masks(self, node_mask, x):
        node_mask = node_mask.squeeze(-1)
        batch_size, n_nodes, n_dims = x.size()

        node_mask_target = torch.zeros(batch_size, self.config.energy_model.max_num_atoms).bool()
        node_mask_target[:, :n_nodes] = node_mask

        node_mask, out_mask = node_mask.to(self.device), node_mask_target.to(self.device)
        
        node_mask = node_mask.unsqueeze(-1)

        
        edge_mask = node_mask.squeeze(-1).unsqueeze(1) * node_mask

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0).to(edge_mask.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
        
        edge_mask = edge_mask.view(batch_size, n_nodes * n_nodes)
        return edge_mask, out_mask


def inverse_normalize(data, mu, std, batch_size, batch):
    mu = mu.reshape(batch_size, 3)
    mu = mu.index_select(0, batch)
    std = std.index_select(0, batch).unsqueeze(-1)
    std = torch.cat([std, std, std], dim=1)
    data = data * std + mu
    return data

def onehot2label(onehot_attr):
    return torch.topk(onehot_attr, 1)[1].squeeze(1) + 1

def clip_norm(vec, limit, p=2):
    norm = torch.norm(vec, dim=-1, p=2, keepdim=True)
    denom = torch.where(norm > limit, limit / norm, torch.ones_like(norm))
    return vec * denom

def sample_center_gravity_zero_gaussian_with_mask(N, device, batch):

    pos_init = torch.randn(N, 3).to(device)
    x, node_mask = to_dense_batch(pos_init, batch, fill_value=0) # x (B, max_num_node, 3), node_mask: (B, max_num_node)
    node_mask = node_mask.unsqueeze(-1) # (B, max_num_node, 1)
    x_masked = x * node_mask
    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean_with_mask(x_masked, node_mask)
    pos_init = x_projected[node_mask.squeeze(-1)]
    return x_projected, pos_init, node_mask

def get_zh_init_with_mask(x, device, batch, norm_biases, norm_values):

    x, node_mask = to_dense_batch(x, batch, fill_value=0)
    node_mask = node_mask.unsqueeze(-1) # (B, max_num_node, 1)
    x = (x.float() - norm_biases) / norm_values
    x_masked = x * node_mask
    return x_masked, node_mask

def remove_mean_with_mask(x, node_mask):

    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x

def l2_square(x, y):
    #input: (batch,c)
    #output: (batch,)
    return (x - y).square()

# Defining some useful util functions.
def expm1(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(x)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)