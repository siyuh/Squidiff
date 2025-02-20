import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import List, Tuple
from . import utils,blocks
from abc import abstractmethod


def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
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

    def __init__(self, time_channels, time_out_channels):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_channels, time_out_channels),
            nn.SiLU(),
            nn.Linear(time_out_channels, time_out_channels),
        )
        self.style = nn.Identity()

    def forward(self, time_emb=None, cond=None):
        if time_emb is not None:
            time_emb = self.time_embed(time_emb)
        style = self.style(cond)
        out = {"style":style,
               'time_emb':time_emb,
               'emb':style,
               
              }
        
        return out
    

@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    num_labels: int
    label_emb_dim: int
    dropout: float = 0.1
    
    def make_model(self):
        return EncoderModel(self)
    
class EncoderModel(nn.Module):
    def __init__(self, conf: EncoderConfig):
        super().__init__()
        
        self.conf = conf
        
        
        self.label_embedding = nn.Linear(1, conf.label_emb_dim, bias=False)
        #self.label_embedding.weight.requires_grad = False
        
        
        
        self.layers = nn.ModuleList()
        self.layers = nn.Sequential(
            nn.Linear(conf.input_dim + conf.label_emb_dim, conf.latent_dim),
            #nn.BatchNorm1d(conf.hidden_dim),
            #nn.ReLU(),
                 
            #nn.Linear(conf.hidden_dim, conf.latent_dim),
            #nn.BatchNorm1d(conf.latent_dim),
            #nn.ReLU(),  
        )
        
        #self.apply(he_init)

        
    def forward(self, x, label = None):
        """
        Apply the model to an input batch

        :param x: Input tensor.
        :param t: Optional timesteps for time embedding.
        :return: Encoded tensor.
        """
        #label = label.unsqueeze(1).float()
        label_emb = self.label_embedding(label)
        h = torch.concat([x,label_emb],axis=1).type(torch.float32)
        h = self.layers(h)
        return h

@dataclass
class NoisePredModelConfig:
    input_dim: int
    out_dim:int
    hidden_dim: int
    latent_dim: int
    
    time_embed_channels: int
    embed_channels : int
    label_emb_dim : int
    num_labels : int
    dropout: float = 0.1

    def make_model(self):
        return NoisePredModel(self)
    
class NoisePredModel(nn.Module):
    def __init__(self,conf: NoisePredModelConfig):
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
            dropout=conf.dropout,
            num_labels = conf.num_labels,
            label_emb_dim = conf.label_emb_dim
            
        ).make_model()
        
        


        
        self.mlp1 = nn.Sequential(
            nn.Linear(conf.input_dim + conf.embed_channels + conf.latent_dim, conf.hidden_dim),
            nn.BatchNorm1d(conf.hidden_dim),
            nn.ReLU())
        
        
        self.mlp2 = nn.Sequential(
            nn.Linear(conf.hidden_dim,conf.latent_dim),
            nn.BatchNorm1d(conf.latent_dim),
            nn.ReLU(),
        )
        
        self.mlp3 = nn.Sequential(
            nn.Linear(conf.latent_dim,conf.hidden_dim),
            nn.BatchNorm1d(conf.hidden_dim),
            nn.ReLU(),
        )

        self.mlp4 = nn.Linear(conf.hidden_dim,conf.out_dim)
        
        #self.apply(he_init)



    def encode(self, x, label):
        return {'cond': self.encoder(x,label)}
    
    
    def forward(self,
                x,
                t,
                x_start=None,
                cond=None,
                label=None,
            ):
        """
        Apply the model to an input batch.

        Args:
            x_start: the original image to encode
            cond: output of the encoder
            noise: random noise (to predict the cond)
        """
        
        #print(label)
        #print(cond)
        
        
        if (cond == None) & (x_start != None):
            tmp = self.encode(x_start,label)
            cond = tmp['cond']
        
        
        
        
        _t_emb = utils.timestep_embedding(t)
        
        res = self.time_embed(time_emb=_t_emb, cond=cond)
        
        enc_time_emb  = res['time_emb']
        enc_cond_emb  = res['emb']
       
        
        
        h = torch.concat([x,enc_time_emb,enc_cond_emb],axis=1)
        h = self.mlp1(h)
        h = self.mlp2(h)
        h = self.mlp3(h)
        h = self.mlp4(h)
        pred = h
        
        out ={"pred":pred,'cond':cond}
        
        return out

    

