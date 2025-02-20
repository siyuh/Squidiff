import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import List, Tuple
from . import utils,blocks
from abc import abstractmethod

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(min(32, channels), channels)

class TimestepEmbedSequential(nn.Sequential, blocks.TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb=None, cond=None, lateral=None):
        
        for layer in self:
            
            if isinstance(layer, blocks.TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:

                x = layer(x)
        return x

    
    
class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features):
        super(ConditionalBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if x.size(0) == 1:
            return x
        else:
            return self.batch_norm(x)


class TimeStyleSeperateEmbed(nn.Module):
    # embed only style
    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_channels, time_out_channels),
            nn.SiLU(),
            nn.Linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None, **kwargs):
        if time_emb is not None:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        out = {"style":style,
               'time_emb':time_emb,
               'emb':style
              }
        
        return out
    

@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    num_layers: int
    dropout: float = 0.1
    def make_model(self):
        return LSTMEncoder(self)

class LSTMEncoder(nn.Module):
    def __init__(self, conf: EncoderConfig):
        super().__init__()
        
        self.conf = conf
        self.lstm = nn.LSTM(conf.input_dim, conf.latent_dim, conf.num_layers, batch_first=True)
        #self.fc = nn.Linear(conf.hidden_dim, conf.latent_dim)
        #self.dropout = nn.Dropout(conf.dropout)
        
    def forward(self, x):
        """
        Apply the model to an input batch.

        :param x: Input tensor.
        :return: Encoded tensor.
        """
        h, _ = self.lstm(x)
        #h = self.dropout(h)
        z_sem = h[:, -1, :]
        #z_sem = self.fc(h[:, -1, :])
        return z_sem
    
@dataclass
class NoisePredModelConfig:
    input_dim: int
    out_dim: int
    hidden_dim: int
    latent_dim: int
    
    time_embed_channels: int
    embed_channels: int
    dropout: float = 0.1
    num_layers: int = 2

    def make_model(self):
        return NoisePredModel(self)

class NoisePredModel(nn.Module):
    def __init__(self, conf: NoisePredModelConfig):
        super().__init__()
        
        self.conf = conf
        
        self.time_embed = TimeStyleSeperateEmbed(
            time_channels=self.conf.time_embed_channels,
            time_out_channels=self.conf.embed_channels,
        )
        
        self.encoder = EncoderConfig(
            input_dim=conf.input_dim,
            hidden_dim=conf.hidden_dim,
            latent_dim=conf.latent_dim,
            num_layers=conf.num_layers,
            dropout=conf.dropout,
        ).make_model()
        
        self.mlp1 = nn.Sequential(
            nn.Linear(conf.input_dim + conf.embed_channels + conf.latent_dim, conf.hidden_dim),
            nn.BatchNorm1d(conf.hidden_dim),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(conf.hidden_dim, conf.hidden_dim),
            nn.BatchNorm1d(conf.hidden_dim),
            nn.ReLU(),
        )
        
        self.mlp3 = nn.Linear(conf.hidden_dim, conf.out_dim)
    def encode(self, x):
        return {'cond': self.encoder(x)}
    def forward(self, x, t, y=None, x_start=None, cond=None, style=None, noise=None, model_kwargs=None):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original data to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        if model_kwargs is None:
            # Ensure x_start has the correct shape for the LSTM
            if x_start.dim() == 2:
                x_start = x_start.unsqueeze(1)  # Add sequence dimension

            tmp = self.encode(x_start)
            cond = tmp['cond']
        else:
            cond = model_kwargs['cond']

        _t_emb = utils.timestep_embedding(t)

        res = self.time_embed(time_emb=_t_emb, cond=cond)

        enc_time_emb = mid_time_emb = dec_time_emb = res['time_emb']
        enc_cond_emb = mid_cond_emb = dec_cond_emb = res['emb']
        style = style or res['style']

        h = torch.cat([x, enc_time_emb, enc_cond_emb], axis=1)
        h = self.mlp1(h)
        h = torch.cat([h], axis=1)
        h = self.mlp2(h)
        h = torch.cat([h], axis=1)
        h = self.mlp3(h)

        pred = h

        out = {"pred": pred, 'cond': cond}

        return out
