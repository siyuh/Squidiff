from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, z_sem):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        
        

class MLPBlock(TimestepBlock):
    """
    Basic MLP block with an optional timestep embedding.
    """

    def __init__(self, input_dim, output_dim, time_embed_dim=None, latent_dim = None):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        self.time_embed_dim = time_embed_dim
        if time_embed_dim is not None:
            self.time_dense = nn.Linear(time_embed_dim, output_dim)
        if latent_dim is not None:
            self.zsem_dense = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x, emb, z_sem):
        
        h = F.silu(self.layer_norm1(self.fc1(x))) 
        if ((emb is not None)&(z_sem is None)):
            h = h + self.time_dense(emb)
        elif ((emb is not None)&(z_sem is not None)):
            h = h + self.time_dense(emb)+self.zsem_dense(z_sem)
        h = F.silu(self.layer_norm2(self.fc2(h)))
        return h
    
    


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, z_sem):
        
        for layer in self:
            
            
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, z_sem)
            else:
                x = layer(x)
        return x

class MLPModel(nn.Module):
    """
    MLP model for single-cell RNA-seq data with timestep embedding.
    """

    def __init__(self, 
                 gene_size, 
                 output_dim, 
                 num_layers, 
                 hidden_sizes=2048,
                 time_pos_dim=2048,
                 num_classes = None,
                 latent_dim=60,
                 use_checkpoint=False,
                 use_fp16 = False,
                 use_scale_shift_norm =False,
                 dropout=0,
                 time_embed_dim=2048,
                 use_encoder=False,
                 use_drug_structure = False,
                 drug_dimension = 1024,
                 comb_num=1,
                 
                ):
        super().__init__()
        
        self.use_encoder = use_encoder
        self.time_embed_dim = time_embed_dim
        self.latent_dim = latent_dim
        self.time_embed = None
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.drug_dimension = drug_dimension
        if use_encoder: 
            self.encoder = EncoderMLPModel(gene_size,self.hidden_sizes, self.num_classes, use_drug_structure, self.drug_dimension, comb_num)
        
        
        if time_embed_dim is not None:
            self.time_embed = nn.Sequential(
                nn.Linear(time_pos_dim, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        if self.use_encoder: 
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim, latent_dim))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        else:
            layers = []
            for _ in range(num_layers):
                layers.append(MLPBlock(hidden_sizes, hidden_sizes, time_embed_dim))
            self.mlp_blocks = TimestepEmbedSequential(*layers)
        
        self.input_layer = nn.Linear(gene_size, hidden_sizes)
        self.output_layer = nn.Linear(hidden_sizes, output_dim)
        
    def forward(self, x, timesteps=None, **model_kwargs):
        
        
        if self.time_embed is not None and timesteps is not None:
            
            emb = self.time_embed(timestep_embedding(timesteps, self.hidden_sizes))
            
        else:
            emb = None
            
        if self.use_encoder: 
            if 'z_mod' in model_kwargs.keys():
                z_sem = model_kwargs['z_mod']
            elif self.num_classes is None:
                z_sem = self.encoder(model_kwargs['x_start'],label = None,drug_dose = model_kwargs['drug_dose'])
            else: 
                z_sem = self.encoder(model_kwargs['x_start'],label = model_kwargs['group'],drug_dose = model_kwargs['drug_dose'])

            h = self.input_layer(x)
            
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        else:
            z_sem = None
            h = self.input_layer(x)
            h = self.mlp_blocks(x=h, emb=emb, z_sem=z_sem)
            h = self.output_layer(h)
        return h

    
class EncoderMLPModel(nn.Module):

    def __init__(self, input_size, hidden_sizes, num_classes=None, use_drug_structure=False, drug_dimension=1024,comb_num=1,output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dtype = th.float16 if use_fp16 else th.float32
        self.drug_dimension = drug_dimension
    
        if num_classes is None:
            l1 = 0
        else: 
            l1 = hidden_sizes
        if use_drug_structure:
            l2 = drug_dimension
        else:
            l2 = 0
        
        self.fc1 = nn.Linear(input_size+l1+l2, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
       
        self.label_embed = nn.Linear(1, hidden_sizes)
    
    def forward(self, x_start, label=None, drug_dose=None):
        
        if label is not None:
            label_emb = self.label_embed(label)
            x_start = th.concat([x_start,label_emb],axis=1)
        
        if drug_dose is not None:
            x_start = th.concat([x_start,drug_dose],axis=1)
            
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.fc3(h)
        return h
    

    
class EncoderMLPModel2(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes=None, output_size=60, dropout=0.1, use_fp16=False):
        super(EncoderMLPModel2, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout = dropout
        self.dtype = th.float16 if use_fp16 else th.float32
        
        self.fc1 = nn.Linear(input_size, hidden_sizes)
        self.bn1 = nn.BatchNorm1d(hidden_sizes)
        self.fc2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.bn2 = nn.BatchNorm1d(hidden_sizes)
        self.fc3 = nn.Linear(hidden_sizes, output_size)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.label_embed = nn.Linear(1, hidden_sizes)  

    def forward(self, x_start, label=None):
        h = x_start.type(self.dtype)
        h = F.relu(self.bn1(self.fc1(h)))
    

        if label is not None:
            label = label.type(self.dtype) 
            label_emb = self.label_embed(label)
            
            h = h + label_emb  # Add label embedding as a residual connection


        h = self.dropout_layer(h)
        h = self.fc3(h)
       
        return h