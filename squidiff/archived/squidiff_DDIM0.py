import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import math
import torch.nn.functional as F
from scipy.interpolate import interp1d
import torch.nn.init as init
import warnings
import logging
from numba.core.errors import NumbaDeprecationWarning
import numpy as np
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)

numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)



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

class SemanticEncoder(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(SemanticEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            ConditionalBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            ConditionalBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def forward(self, x):
        z_sem = self.model(x)
        return z_sem

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        
class SquidModel(nn.Module):
    def __init__(self, input_dim, z_sem_dim, time_emb_dim=32, hidden_dim=512, n_steps=100, device='cpu',model_var_type='fixedsmall',reverse_steps=100):
        super().__init__()
        
        self.n_steps = n_steps
        self.device = device
        self.model_var_type = model_var_type
        self.reverse_steps = reverse_steps
        
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        
        self.MLP = nn.Sequential(
            nn.Linear(input_dim + z_sem_dim + time_emb_dim, hidden_dim),
            ConditionalBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            ConditionalBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self.betas = torch.linspace(0.0001, 0.02, self.n_steps).to(self.device)
        
        self.alphas = (1. - self.betas).to(self.device)
    
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.zeros(1).to(self.device)], dim=0
        )

        self.semantic_encoder = SemanticEncoder(input_dim, z_sem_dim, hidden_dim)



    def predict_noise(self, x, t, z_sem):
        """
        reverse diffusion process
        predict epsilon_theta(x_t, t) to infer the p(x_t-1|x_t,z_sem)
        """
        
        t_emb = self.time_emb(t)
        combined_input = torch.cat([x, t_emb, z_sem], dim=-1)
        
        return self.MLP(combined_input)

    def diffuse(self, x_0, t, noise):
        
        """
        forward diffusion process in DDIM
        q(x_t|x_0) = N(sqrt_alpha_t * x0, (1-alpha_t)*I)
        """
        x_0, t, noise = x_0.to(self.device), t.to(self.device), noise.to(self.device)
        
        a = self.alphas_cumprod.gather(0, t).unsqueeze(-1)
        
        return torch.sqrt(a) * x_0 + torch.sqrt(1-a) * noise


    def p_loss(self, x_0, t):
        """
        Compute the loss for the diffusion process.

        Args:
            x_0: Original data.
            t: Timesteps.

        Returns:loss
        """
        x_start = x_0.unsqueeze(0).expand(t.size(0), -1, -1).reshape(-1, x_0.size(-1))
        t_reshaped = t.reshape(-1)
        
        noise = torch.randn_like(x_start)
        
        # semantic encoder 
        z_sem = self.semantic_encoder(x_start)
        
        # diffuse  == q_sample
        x_t = self.diffuse(x_start, t_reshaped, noise)
        
        noise_computed = self.predict_noise(x_t, t_reshaped, z_sem)
    

        # Compute L2 loss to making the predicted noise to the real noise
        mse_loss = F.mse_loss(noise, noise_computed)

        return mse_loss, z_sem, x_t
    
    
    def get_timestep_sequence(num_timesteps, timesteps, skip_type):
        if skip_type == "uniform":
            skip = num_timesteps // timesteps
            seq = range(0, num_timesteps, skip)
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt(num_timesteps * 0.8), timesteps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        return seq

    def compute_alpha(self, t):
        beta = torch.cat([torch.zeros(1).to(self.device), self.betas], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1)
        return a

    def reconstruct(self, x_T, z_sem,seq, eta = 0 ):
        """
        Reconstruct the original data x_0 from noisy data x_T using reverse diffusion.
        """
        #x = x_T.float()
        
        with torch.no_grad(): 
            n = x_T.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs=[x_T]
            
            for i, j in zip(reversed(seq),reversed(seq_next)):
                t = torch.full((n,), i, dtype=torch.long, device=self.device)
                next_t = torch.full((n,), j, dtype=torch.long, device=self.device)
                
                at = self.compute_alpha(t)
                at_next = self.compute_alpha(next_t)
                
                xt = xs[-1]
                
                epsilon_theta = self.predict_noise(xt, t, z_sem)
                x0_t = (xt - epsilon_theta * torch.sqrt(1 - at)) / torch.sqrt(at)
                x0_preds.append(x0_t)
                #print(at.shape)
                #print((torch.sqrt(1 - at)).shape)
                #print((epsilon_theta * torch.sqrt(1 - at)).shape)
            
                xt_next = x0_t * torch.sqrt(at_next) + epsilon_theta * torch.sqrt(1 - at_next)
                
                xs.append(xt_next)
                
        return xs[-1], x0_preds
    
    
    def stochastic_encode(self, x_0, T, z_sem):
        device = x_0.device
        x_t = x_0

        for t in range(0, T-1):
            t_tensor = torch.tensor(t, dtype=torch.long, device=device)
            
            a = self.alphas_cumprod.gather(0, t_tensor).unsqueeze(-1)
            a_p1 = self.alphas_cumprod_next.gather(0, t_tensor).unsqueeze(-1)
            
            # Get predicted noise
            epsilon_theta_t = self.predict_noise(
                x_t, 
                torch.full((x_t.size(0),), t, dtype=torch.long, device=device), 
                z_sem
            )
            
            # Get f_theta
            f_theta_t = (x_t - torch.sqrt(1 - a) * epsilon_theta_t) / torch.sqrt(a)

            x_t = torch.sqrt(a_p1) * f_theta_t + torch.sqrt(1 - a_p1) * epsilon_theta_t
    
        return x_t
    


