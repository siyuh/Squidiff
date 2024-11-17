import torch.nn as nn
import torch
import math
from dataclasses import dataclass
from typing import Tuple

@dataclass
class EncoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    num_layers: int
    dropout: float = 0.1

    def make_model(self):
        return EncoderModel(self)


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


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    

class SemanticEncoder(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(SemanticEncoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ConditionalBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            #ConditionalBatchNorm1d(output_dim),
        )
        
    def forward(self, x):
        z_sem = self.network(x)
        return z_sem
    
class NoisePredictionNetwork(nn.Module):
    def __init__(self, input_dim, z_sem_dim, time_emb_dim, hidden_dim):
        super(NoisePredictionNetwork, self).__init__()
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.fc1 = nn.Linear(input_dim  + time_emb_dim + z_sem_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim , hidden_dim)
        self.fc3 = nn.Linear(hidden_dim , input_dim)
        self.relu = nn.ReLU()
        #self.batchnorm1 = ConditionalBatchNorm1d(hidden_dim)
        #self.batchnorm2 = ConditionalBatchNorm1d(hidden_dim)
        self.semantic_encoder = SemanticEncoder(input_dim, z_sem_dim, hidden_dim)
        #self.label_emb = nn.Embedding(4, time_emb_dim)

    def forward(self, x, t, z_sem = None, y=None):
        
        #if z_sem==None:
        #    z_sem = self.semantic_encoder(x)

        t_emb = self.time_emb(t)
        t_emb = t_emb# + z_sem 
        
        x = torch.cat([x, t_emb, z_sem], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
