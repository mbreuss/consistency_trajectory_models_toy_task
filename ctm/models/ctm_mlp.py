from multiprocessing.sharedctypes import Value
import einops

import torch
from torch import DictType, nn
from omegaconf import DictConfig
import numpy as np 


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=0.02):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class MLPNetwork(nn.Module):
    """Simple multi-layer perceptron network."""
    def __init__(self, input_dim, hidden_dim=100, num_hidden_layers=1, output_dim=1, dropout_rate=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)] + \
                 [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)] + \
                 [nn.Linear(hidden_dim, output_dim)]
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.Mish()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation and dropout to all but the last layer
                x = self.act(x)
                x = self.dropout(x)
        return x

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)
    
    def get_params(self):
        return self.layers.parameters()


class ConsistencyTrajectoryNetwork(nn.Module):
    def __init__(self, x_dim, hidden_dim, time_embed_dim, cond_dim, cond_mask_prob,
                 num_hidden_layers=1, output_dim=1, dropout_rate=0.0, cond_conditional=True):
        super().__init__()
        self.embed_t = GaussianFourierProjection(time_embed_dim)
        self.embed_s = GaussianFourierProjection(time_embed_dim)
        self.cond_mask_prob = cond_mask_prob
        self.cond_conditional = cond_conditional
        input_dim = time_embed_dim * 2 + x_dim + (cond_dim if cond_conditional else 0)
        self.mlp = MLPNetwork(input_dim, hidden_dim, num_hidden_layers, output_dim, dropout_rate)

    def forward(self, x, cond, t, s):
        t = t.view(-1, 1)
        s = s.view(-1, 1)

        embed_t = self.embed_t(t).squeeze(1)
        embed_s = self.embed_s(s).squeeze(1)
        if embed_s.shape[0] != x.shape[0]:
            embed_s = einops.repeat(embed_s, '1 d -> (1 b) d', b=x.shape[0])
        if embed_t.shape[0] != x.shape[0]:
            embed_t = einops.repeat(embed_t, '1 d -> (1 b) d', b=x.shape[0])
        x = torch.cat([x, cond, embed_s, embed_t], dim=-1) if self.cond_conditional else torch.cat([x, embed_s, embed_t], dim=-1)
        return self.mlp(x)


def rearrange_for_batch(x, batch_size):
    """Utility function to repeat the tensor for the batch size."""
    return x.expand(batch_size, -1)

    def get_params(self):
        return self.parameters()