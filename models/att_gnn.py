
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
from einops import repeat
import torch
from torch import nn


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.GELU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        self.encoder = MLP([12] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts):
        # inputs = [kpts]
        return self.encoder(kpts)


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class transformer_mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        # self.mlp = transformer_mlp(feature_dim, feature_dim*4, feature_dim)
        # nn.init.constant_(self.mlp[-1].bias, 0.0)
        # self.norm1 = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        message = self.attn(x, source, source)

        return message

        # x = x + message
        # x = x.permute(0,2,1)
        # x = self.norm1(x)

        # x = self.mlp(x)
        # x = x.permute(0,2,1)

        # return x
        # print (x.shape, message.shape)
        # return self.mlp(torch.cat([x, message], dim=1))

class MemoryGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers_grasp = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        
        self.layers_seed = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names) - 1)])

        self.layers_seed.append(None)

        self.names = layer_names

        self.mlp_layers_grasp = nn.ModuleList([
            transformer_mlp(feature_dim, feature_dim*4, feature_dim)
            for _ in range(len(layer_names))])

        self.mlp_layers_seed = nn.ModuleList([
            transformer_mlp(feature_dim, feature_dim*4, feature_dim)
            for _ in range(len(layer_names) - 1)])
        
        self.mlp_layers_seed.append(None)
        # self.fine = fine
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # self.temporal_embedding = nn.Parameter(torch.zeros((256,9)), requires_grad=True)

        self.norm3 = nn.LayerNorm(feature_dim)
        self.norm4 = nn.LayerNorm(feature_dim)


        # self.mlp = transformer_mlp(feature_dim, feature_dim*4, feature_dim)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        
        # desc0 = desc0 + self.temporal_embedding[None, :, :].to(desc0.dtype)
        # desc1 = desc1 + self.temporal_embedding[None, :, :].to(desc1.dtype)

        bn, C, M = desc0.shape

        for layer_grasp, layer_seed, mlp_grasp, mlp_seed, name in zip(self.layers_grasp, self.layers_seed, self.mlp_layers_grasp, self.mlp_layers_seed,  self.names):

            if name == 'cross':
                # desc0 = desc0.view(-1 ,C, M, 1).permute(0, 2, 1, 3).contiguous().view(-1, C, 1)
                # desc1 = desc1.permute(0, 3, 1, 2).contiguous().view(-1, C, Ns)
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                # desc1 = desc1.view(-1, C, Ns*M)
                src0, src1 = desc0, desc1
            
            if layer_seed is not None:
                assert mlp_seed is not None
                delta0, delta1 = layer_grasp(desc0, src0), layer_seed(desc1, src1)

                desc0, desc1 = self.norm1((desc0 + delta0).permute(0,2,1)), self.norm2((desc1 + delta1).permute(0,2,1))

                delta0, delta1 = mlp_grasp(desc0), mlp_seed(desc1)
                desc0, desc1 = self.norm3((desc0 + delta0)).permute(0,2,1), self.norm4((desc1 + delta1)).permute(0,2,1)
            
            else:
                assert mlp_seed is None

                delta0 = layer_grasp(desc0, src0)

                desc0 = self.norm1((desc0 + delta0).permute(0,2,1))

                delta0 = mlp_grasp(desc0)
                desc0 = self.norm3((desc0 + delta0)).permute(0,2,1)



        #     if name != 'cross':
        #         desc1 = desc1.view(-1 ,C, Ns, M)
        #     else:
        #         desc0 = desc0.view(_, M, C, 1).squeeze(-1).permute(0, 2, 1).contiguous()
        #         desc1 = desc1.view(_, M, C, Ns).permute(0,2,3,1).contiguous()
        # # print ('------')
        return desc0, desc1

class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

        self.mlp_layers = nn.ModuleList([
            transformer_mlp(feature_dim, feature_dim*4, feature_dim)
            for _ in range(len(layer_names))])
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # self.norm3 = nn.LayerNorm(feature_dim)
        # self.norm4 = nn.LayerNorm(feature_dim)


        # self.mlp = transformer_mlp(feature_dim, feature_dim*4, feature_dim)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        for layer, mlp, name in zip(self.layers, self.mlp_layers,  self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)

            
            desc0, desc1 = self.norm1((desc0 + delta0).permute(0,2,1)), self.norm1((desc1 + delta1).permute(0,2,1))

            delta0, delta1 = mlp(desc0), mlp(desc1)
            desc0, desc1 = self.norm2((desc0 + delta0)).permute(0,2,1), self.norm2((desc1 + delta1)).permute(0,2,1)



        return desc0, desc1


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1

