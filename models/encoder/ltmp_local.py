from torch import nn, einsum
import torch

import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from inspect import isfunction

def get_pair_dis_one_hot(d1, d2, bin_size=2, bin_min=-1, bin_max=30):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    pair_dis = torch.cdist(d1, d2, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    return pair_dis_one_hot

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)
        
class LTMPLocal(nn.Module):
    def __init__(self, config):
        super(LTMPLocal, self).__init__()
        self.config = config
        # target and ligand embedding
        self.l_pair_embedding = nn.Linear(16, config.ligand_dim)
        if self.config.local.use_direct:
            self.protein_pair_embedding = nn.Linear(16, config.hidden_dim)
        if self.config.local.use_undirect:
            self.inter_pair_embedding = nn.Linear(16, config.hidden_dim)
        self.ligandEmbedding = LigandEmbedding(config.target_dim, config.ligand_dim)
        self.targetEmbedding = TargetEmbedding(config.in_dim, config.ligand_dim, config.target_dim)

        # main trunk modules

        self.net = Evoformer(config)
        self.ligand_logits = nn.AdaptiveAvgPool2d((1, config.out_dim))
        self.target_logits = nn.AdaptiveAvgPool2d((1, config.out_dim))

    def forward(self, h_l, h_t):

        h_l_x, l_mask = to_dense_batch(h_l.x, h_l.batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(h_t['embedding_1'], h_t['batch'], fill_value=0)
        h_l_pos, _ = to_dense_batch(h_l.pos, h_l.batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(h_t['xyz'], h_t['batch'], fill_value=0)
        
        l_pair = get_pair_dis_one_hot(h_l_pos, h_l_pos, bin_size=1, bin_min=-0.5, bin_max=15)
        l_pair = self.l_pair_embedding(l_pair.float())
        if self.config.local.use_direct:
            protein_pair = get_pair_dis_one_hot(h_t_pos, h_t_pos, bin_size=2, bin_min=-1, bin_max=15)
            protein_pair = self.protein_pair_embedding(protein_pair.float())
        else:
            protein_pair = None
        if self.config.local.use_undirect:
            inter_pair = get_pair_dis_one_hot(h_t_pos, h_l_pos, bin_size=1, bin_min=-0.5, bin_max=15)
            inter_pair = self.inter_pair_embedding(inter_pair.float())
        else:
            inter_pair = None

        device = h_l.x.device

        # embed target and ligand
        l_x, t_x = self.targetEmbedding(h_l_x, h_t_x) #l_x: (B, L, H) t_x: (B, T, L, H)
        if self.config.use_dist_embed:
            l_x, l_x_mask = self.ligandEmbedding(l_x, l_mask, l_pair=l_pair) #ligandwise embedding: B*L*L*d_ligand (2*138*138*128) mask: B, L, L
        else:
            l_x, l_x_mask = self.ligandEmbedding(l_x, l_mask, l_pair=None)
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        C_mask = t_mask.view(B, N_t, 1) & l_mask.view(B, 1, N_l) 
        # trunk
        # ligand (B, L, L, d_ligand), target (B, N, L, d_target)
       
        l_x_out, t_x_out = self.net(
            l_x,
            t_x,
            l_pair,
            protein_pair,
            inter_pair,
            mask=l_x_mask,  # ligand_mask
            target_mask=C_mask
        )

        l_feat = self.ligand_logits(l_x_out).squeeze(2)[l_mask]
        t_feat = self.target_logits(t_x_out).squeeze(2)[t_mask]
        z_out = torch.cat([l_feat, t_feat], dim=0)

        return z_out
        
# embedding classes

class TargetEmbedding(nn.Module):
    def __init__(self, in_dim, ligand_dim, target_dim):
        super().__init__()
        self.lig_emb = nn.Linear(in_dim, ligand_dim)
        self.tar_emb = nn.Linear(in_dim, target_dim)

    def forward(self, h_l_x, h_t_x):
        
        lig_embed = self.lig_emb(h_l_x)
        target_embed = self.tar_emb(h_t_x)
        # add single representation to target representation target=target+ligand
        target_embed = rearrange(target_embed, 'b n d -> b n () d') + rearrange(lig_embed, 'b n d -> b () n d')
        return lig_embed, target_embed


class LigandEmbedding(nn.Module):
    def __init__(self, target_dim, ligand_dim):
        super().__init__()
        self.to_ligandwise_repr = nn.Linear(target_dim, 2 * ligand_dim)
 
    def forward(self, h_l_x, h_l_x_mask, l_pair):
        x_left, x_right = self.to_ligandwise_repr(h_l_x).chunk(2, dim=-1)  # B*L*d_ligand 2*138*128

        # create ligand-wise residue embeds x: B*L*L*d_ligand 2*138*138*128
        x = rearrange(x_left, 'b i d -> b i () d') + rearrange(x_right,
                                                               'b j d-> b () j d')
        if l_pair is not None:
            ligand_embed = x + l_pair  # ligandwise embedding: B*L*L*d_ligand (2*138*138*128)
        else:
            ligand_embed = x
        ligand_mask = rearrange(h_l_x_mask, 'b i -> b i ()') * rearrange(h_l_x_mask, 'b j -> b () j') if exists(h_l_x_mask) else None
        return ligand_embed, ligand_mask

class TriangleDistToZ(nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.layernorm_c = nn.LayerNorm(c)

        self.gate_linear1 = nn.Linear(embedding_channels, c)
        self.gate_linear2 = nn.Linear(embedding_channels, c)
        self.gate_linear3 = nn.Linear(embedding_channels, c)

        self.linear1 = nn.Linear(embedding_channels, c)
        self.linear2 = nn.Linear(embedding_channels, c)
        self.linear3 = nn.Linear(embedding_channels, c)

        self.ending_gate_linear = nn.Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = nn.Linear(c, embedding_channels)
    def forward(self, z, l_pair, protein_pair, inter_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is l dim.
        # protein_pair, l_pair are pairwise distances
        z = self.layernorm(z)
        z_mask = z_mask.unsqueeze(-1)
        block = None
        if protein_pair is not None:
            protein_pair = self.layernorm(protein_pair)
            l_pair = self.layernorm(l_pair)
            ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask
            ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask
            protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
            l_pair = self.gate_linear1(l_pair).sigmoid() * self.linear1(l_pair)
        if inter_pair is not None:
            inter_pair = self.layernorm(inter_pair)
            ab3 = self.gate_linear3(z).sigmoid() * self.linear3(z) * z_mask
            inter_pair = self.gate_linear1(inter_pair).sigmoid() * self.linear1(inter_pair)

        g = self.ending_gate_linear(z).sigmoid()
        if protein_pair is not None:
            block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
            block2 = torch.einsum("bikc,bjkc->bijc", ab2, l_pair)
            block = block1 + block2
            
        if inter_pair is not None:
            block3 = torch.einsum("bijc,bijc->bijc", ab3, inter_pair)
            if block is None:
                block = block3
            else:
                block = block + block3

        z = g * self.linear_after_sum(self.layernorm_c(block)) * z_mask

        return z
# feed forward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        init_zero_(self.net[-1])

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)


# attention

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            gating=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, mask=None, attn_bias=None, context=None, context_mask=None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # scale

        q = q * self.scale

        # query / key similarities

        
        dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for ligandwise to target attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device=device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2],
                                                                                                 device=device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention

        dots = dots - dots.max(dim=-1, keepdims=True).values
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        gates = self.gating(x)
        out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out


class AxialAttention(nn.Module):
    def __init__(
            self,
            heads,
            need_ligand=False,
            need_target=False,
            target_dim=None,
            ligand_dim=None,
            row_attn=True,
            col_attn=True,
            accept_edges=False,
            global_query_attn=False,
            **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'
        self.target_dim=target_dim
        self.ligand_dim=ligand_dim
        self.need_ligand=need_ligand
        self.need_target = need_target

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn


        self.target_norm = nn.LayerNorm(target_dim) if need_target else None
        self.ligand_norm = nn.LayerNorm(ligand_dim) if need_ligand else None
        self.ligand_attn = Attention(dim=ligand_dim, heads=heads, **kwargs) if need_ligand else None
        self.target_attn = Attention(dim=target_dim, heads=heads, **kwargs) if need_target else None

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(ligand_dim, heads, bias=False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

    def forward(self, target=None, ligand=None, edges=None, mask=None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'
        
        b, h, w, d = target.shape if exists(target) else ligand.shape # 2*430*138*256

        # axial attention

        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'


        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges) # b: 2*8*138*138
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x=axial_dim) # (138*2=276)*8*138*138

        

        if self.need_target:
            if self.need_ligand:
                # ligand2target
                target = self.target_norm(target)
                target = rearrange(target, input_fold_eq)
                target = self.target_attn(target, mask=mask, attn_bias=attn_bias)
                target = rearrange(target, output_fold_eq, h=h, w=w)
                return target
            else:
                #target2target
                target = self.target_norm(target)
                target = rearrange(target, input_fold_eq)
                target = self.target_attn(target, mask=mask, attn_bias=None)
                target = rearrange(target, output_fold_eq, h=h, w=w)
                return target
        else:
            #ligand2ligand
            ligand = self.ligand_norm(ligand)
            ligand = rearrange(ligand, input_fold_eq)
            ligand = self.ligand_attn(ligand, mask=mask, attn_bias=attn_bias)
            ligand = rearrange(ligand, output_fold_eq, h=h, w=w)
            return ligand


class TriangleMultiplicativeModule(nn.Module):
    def __init__(
            self,
            *,
            dim,
            hidden_dim=None,
            mix='ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate #(B*L*L*d_hidden)
        return self.to_out(out) #(B*L*L*d_ligand)


# evoformer blocks

class OuterMean(nn.Module):
    def __init__(
            self,
            target_dim,
            ligand_dim,
            hidden_dim=None,
            eps=1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(target_dim)
        hidden_dim = default(hidden_dim, target_dim)

        self.left_proj = nn.Linear(target_dim, hidden_dim)
        self.right_proj = nn.Linear(target_dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, ligand_dim)

    def forward(self, x, mask=None):
        x = self.norm(x) #target
        left = self.left_proj(x) #(2*430*138*32)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d') ##(2*430*138*138*32)

        if exists(mask):
            # masked mean, if there are padding in the rows of the target
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.)
            outer = outer.mean(dim=1) / (mask.sum(dim=1) + self.eps)
        else:
            outer = outer.mean(dim=1) #(2*138*138*32)

        return self.proj_out(outer) #(2*138*138*128)


class ligandwiseAttentionBlock(nn.Module):
    def __init__(
            self,
            ligand_dim,
            heads,
            dim_head,
            dropout=0.,
            global_column_attn=False
    ):
        super().__init__()

        self.triangle_attention_outgoing = AxialAttention(ligand_dim=ligand_dim, heads=heads, dim_head=dim_head, row_attn=True,
                                                          col_attn=False, accept_edges=True, need_ligand=True, need_target=False)
        self.triangle_attention_ingoing = AxialAttention(ligand_dim=ligand_dim, heads=heads, dim_head=dim_head, row_attn=False,
                                                         col_attn=True, accept_edges=True, need_ligand=True, need_target=False,
                                                         global_query_attn=global_column_attn)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=ligand_dim, mix='outgoing', hidden_dim=32)
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=ligand_dim, mix='ingoing', hidden_dim=32)

    def forward(
            self,
            x, # ligand from target2ligand
            ligand_in, #ligand_into the evoformer block
            mask=None,
            target_mask=None
    ):
        x = x + ligand_in
        x = self.triangle_multiply_outgoing(x, mask=mask) + x
        x = self.triangle_multiply_ingoing(x, mask=mask) + x #ligand 2*138*138*128
        x = self.triangle_attention_outgoing(ligand=x, edges=x, mask=mask) + x
        x = self.triangle_attention_ingoing(ligand=x, edges=x, mask=mask) + x
        return x


class targetAttentionBlock(nn.Module):
    def __init__(
            self,
            dim,
            heads,
            dim_head,
            dropout=0.
    ):
        super().__init__()
        self.row_attn = AxialAttention(target_dim=dim, heads=heads, dim_head=dim_head, row_attn=True, col_attn=False, need_ligand=False, need_target=True)
        self.col_attn = AxialAttention(target_dim=dim, heads=heads, dim_head=dim_head, row_attn=False, col_attn=True, need_ligand=False, need_target=True)

    def forward(
            self,
            x,  # target
            mask=None
    ):
        x = self.row_attn(target=x, mask=mask) + x
        x = self.col_attn(target=x, mask=mask) + x # target B*N*L*d_target
        return x


class ligandBiastarget(nn.Module):
    def __init__(
            self,
            target_dim,
            ligand_dim,
            heads,
            dim_head,
            dropout=0.
    ):
        super().__init__()
        self.ligand_bias_attn = AxialAttention(ligand_dim=ligand_dim, target_dim=target_dim, heads=heads, dim_head=dim_head, row_attn=True, col_attn=False, need_ligand=True, need_target=True, accept_edges=True)

    def forward(
            self,
            ligand,
            target,
            mask=None,
    ):
        target = self.ligand_bias_attn(target=target, ligand=ligand, mask=mask, edges=ligand) + target
        return target

# main evoformer class

class EvoformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        target_dim = config.target_dim
        ligand_dim = config.ligand_dim
        heads = config.heads
        dim_head = config.dim_head
        attn_dropout = config.attn_dropout
        ff_dropout = config.ff_dropout
        global_column_attn = config.global_column_attn
        self.dist_to_z = TriangleDistToZ(embedding_channels=config.target_dim, c=config.hidden_dim)
        self.layer = nn.ModuleList([
            ligandBiastarget(target_dim=target_dim, ligand_dim=ligand_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
            targetAttentionBlock(dim=target_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout),
            FeedForward(dim=target_dim, dropout=ff_dropout),
            OuterMean(target_dim=target_dim, ligand_dim=ligand_dim, hidden_dim=32),
            ligandwiseAttentionBlock(ligand_dim=ligand_dim, heads=heads, dim_head=dim_head, dropout=attn_dropout,
                                   global_column_attn=global_column_attn),
            FeedForward(dim=ligand_dim, dropout=ff_dropout)
        ])

    def forward(self, target, ligand, t_pair, l_pair, inter_pair, mask=None, target_mask=None):
        ligand2target, target2target, target_ff, target2ligand, ligand2ligand, ligand_ff = self.layer
        ligand_in=ligand # [B, L, L, D]
        
        # 1. ligand2target
        target = self.dist_to_z(target, l_pair, t_pair, inter_pair, target_mask)
        target = ligand2target(ligand=ligand, target=target, mask=target_mask) #[B, T, L, D]

        # 2. target2target target attention and transition
        target = target2target(target, mask=target_mask)
        target = target_ff(target) + target

        # 3. target2ligand
        ligand = target2ligand(target, mask=target_mask) # [B, L, L, D]

        # 4. ligand2ligand ligandwise attention and transition
        ligand = ligand2ligand(ligand, mask=mask, ligand_in=ligand_in, target_mask=target_mask)
        ligand = ligand_ff(ligand) + ligand

        return ligand, target, mask, target_mask


class Evoformer(nn.Module):
    def __init__(
            self,
            config
    ):
        super().__init__()
        
        self.depth = config.depth
        self.evo_blo = EvoformerBlock(config)
    def forward(
            self,
            l_x, # ligand (B*L*L*d_ligand) 2*138*138*128
            t_x, # target (B, N, L, d_target) 2*430*138*128
            l_pair,
            t_pair,
            inter_pair,
            mask=None,  # ligand_mask
            target_mask=None
    ):
        for _ in range(self.depth):
            l_x, t_x, *_ = self.evo_blo(t_x, l_x, t_pair, l_pair, inter_pair, mask, target_mask)
        
        return l_x, t_x
