import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import warnings
import logging
import numpy as np
import json
import math

from numba.core.errors import NumbaDeprecationWarning
from scipy.interpolate import interp1d


warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)


from . import datasets

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
            nn.Linear(hidden_dim, 256),
            ConditionalBatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
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
    
    
class NoisePredictionNetwork(nn.Module):
    
    def __init__(self, input_dim, z_sem_dim, time_emb_dim, hidden_dim):
        super(NoisePredictionNetwork, self).__init__()
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        self.fcn  = nn.Sequential(
            nn.Linear(input_dim + z_sem_dim + time_emb_dim, hidden_dim),
            ConditionalBatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t, z_sem):
        t_emb = self.time_emb(t)
        x = torch.cat([x, t_emb, z_sem], dim=1)
        x = self.fcn(x)
        return x
        
class SquidModel(nn.Module):
    def __init__(self, input_dim, 
                 z_sem_dim, 
                 n_train_epochs=1500,
                 n_steps=1000, 
                 time_emb_dim=30, 
                 hidden_dim=512,
                 batch_size=16,
                 num_workers=1,
                 device='cpu',model_var_type='fixedsmall',reverse_steps=100):
        super().__init__()
        
        self.T = n_steps
        self.batch_size = batch_size
        self.n_train_epochs = n_train_epochs
        self.device = device
        self.model_var_type = model_var_type
        self.reverse_steps = reverse_steps
        self.num_workers = num_workers
        
        
        self.time_emb = SinusoidalPositionEmbeddings(time_emb_dim)
        
        self.MLP = NoisePredictionNetwork(input_dim, z_sem_dim, time_emb_dim, hidden_dim)
        
        self.semantic_encoder = SemanticEncoder(input_dim, z_sem_dim, hidden_dim)

        self.betas = torch.linspace(0.0001, 0.02, self.T+1).to(self.device)
        
        self.num_timesteps = int(self.betas.shape[0])
        
        self.alphas = 1. - self.betas
    
        self.alphas_cumprod = self.alphas.cumprod(dim=0)
        
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0], device=self.device)])
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]]))
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
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
        z_sem = self.semantic_encoder(x_start)
        noise = torch.randn_like(x_start)
        x_t = self.diffuse(x_start, t_reshaped, noise)
        noise_computed = self.predict_noise(x_t, t_reshaped, z_sem)
        loss = F.mse_loss(noise, noise_computed)
        return loss, z_sem, x_t

    def predict_noise(self, x, t, z_sem):
        """
        reverse diffusion process
        predict epsilon_theta(x_t, t) to infer the p(x_t-1|x_t,z_sem)
        """

        return self.MLP(x, t, z_sem)   
    
    def get_timestep_sequence(self, skip_type='uniform'):
        if skip_type == "uniform":
            skip = (self.T+1) // self.reverse_steps
            seq = range(0, self.T+1, skip)
            seq = [int(s) for s in list(seq)]
        elif skip_type == "quad":
            seq = (np.linspace(0, np.sqrt((self.T+1) * 0.8), self.reverse_steps) ** 2)
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        return seq
    
                
                


    def diffuse(self, x_0, t, noise):
        
        """
        forward diffusion process in DDIM
        q(x_t|x_0) = N(sqrt_alpha_t * x0, (1-alpha_t)*I)
        """
        x_0, t, noise = x_0.to(self.device), t.to(self.device), noise.to(self.device)
        
        a = self.alphas_cumprod.gather(0, t).unsqueeze(-1)
        
        return torch.sqrt(a) * x_0 + torch.sqrt(1-a) * noise


    

    def ddim_steps(self, x_T, z_sem, seq):

        with torch.no_grad(): 
            n = x_T.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs=[x_T]
            

            for i, j in zip(reversed(seq),reversed(seq_next)):
                
                if j>(-1):
                    
                    t = torch.full((n,), i, dtype=torch.long, device=self.device)
                    next_t = torch.full((n,), j, dtype=torch.long, device=self.device)

                    at = self.alphas_cumprod[i]
                    at_next = self.alphas_cumprod[j]

                    xt = xs[-1]

                    #print(at)
                    epsilon_theta = self.predict_noise(xt, t, z_sem)
                    x0_t = (xt - epsilon_theta * torch.sqrt(1 - at)) / torch.sqrt(at)
                    x0_preds.append(x0_t)

                    xt_next = x0_t * torch.sqrt(at_next) + epsilon_theta * torch.sqrt(1 - at_next)

                    xs.append(xt_next)
            
        return xs, x0_t  
    
    
    def ddpm_steps(self, x, z_sem, seq):
        
        with torch.no_grad(): 
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xs = [x]
            x0_preds = []
            
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = torch.full((n,), i, dtype=torch.long, device=self.device)
                next_t = torch.full((n,), j, dtype=torch.long, device=self.device)

                at = self.alphas_cumprod[t]
                at_next = self.alphas_cumprod[next_t]
                
                beta_t = 1 - at / at_next
                x = xs[-1].to('cuda')

                e = self.predict_noise(x, t, z_sem)

                x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
                #x0_from_e = torch.clamp(x0_from_e, -1, 1)
                x0_preds.append(x0_from_e.to('cpu'))
                mean_eps = (
                    (at_next.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - at_next)) * x
                ) / (1.0 - at)

                mean = mean_eps
                noise = torch.randn_like(x)
                
                mask = 1 - (t == 0).float()
                mask = mask.view(-1,1)
                logvar = beta_t.log()
                
                sample = mean + mask * torch.exp(0.5 * logvar) * noise
                
                xs.append(sample.to('cpu'))
                
        return xs, x0_preds
        

    
 
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t,
                                     x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t,
                                     x_t.shape) * eps)
    
    
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) *
            x_start +
            _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) *
            x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t,
                                                  x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] ==
                posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    
    def p_mean_variance(self, x, t, z_sem,clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x, x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        

        B, C = x.shape[:2]
     
        assert t.shape == (B, )

        model_forward = self.predict_noise(x=x, t=t, z_sem=z_sem)
        model_output = model_forward

        model_variance = _extract_into_tensor(self.posterior_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t)

        assert (model_mean.shape == model_log_variance.shape ==
                pred_xstart.shape == x.shape)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }


    def ddim_sample(self, x, t, z_sem, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, eta=0.0):
        """
        Perform one step of DDIM sampling.
        """
        out = self.p_mean_variance(
            x,
            t,
            z_sem,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        #noise 
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape)
               * x - out["pred_xstart"]) / _extract_into_tensor(
               self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_next) +
                     torch.sqrt(1 - alpha_bar_next) * eps)

        if eta > 0:
            noise = torch.randn_like(img)
            mean_pred = mean_pred + eta * torch.sqrt(1 - alpha_bar_next) * noise

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

        
    
    def ddim_sample_loop_progressive(self, z_sem,shape=None, noise=None, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, device=None, progress=False, eta=0.0):
        
        if device is None:
            device = next(self.parameters()).device
        if noise is not None:
            x = noise
        else:
            assert isinstance(shape, (tuple, list))
            x = torch.randn(*shape, device=device)
        indices = list(range(self.T+1))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            if isinstance(model_kwargs, list):
                _kwargs = model_kwargs[i]
            else:
                _kwargs = model_kwargs

            t = torch.tensor([i] * len(x), device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    x,
                    t,
                    z_sem,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=_kwargs,
                    eta=eta,
                )
                out['t'] = t
                yield out
                x = out["sample"]   
                
                
                
    def ddim_sample_loop(self, 
                         z_sem,
                         shape=None,
                         noise=None, 
                         clip_denoised=True, 
                         denoised_fn=None, 
                         cond_fn=None, 
                         model_kwargs=None, 
                         device=None, 
                         progress=False, 
                         eta=0.0):
        
        final = None
        for sample in self.ddim_sample_loop_progressive(
                z_sem,
                shape=shape,
                noise=noise,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return final['sample']    
        
    def sample_seq(self, x_T, z_sem, eta = 0 , sample_type ='DDIM'):
        """
        the reverse process: sample the single cell data x_0 from noisy data x_T and z_sem
        """
        seq = self.get_timestep_sequence()
        
        if sample_type == 'DDIM':
            
            #x= self.ddim_steps(z_sem, noise =  x_T,)
            x = self.ddim_steps(x_T, z_sem, seq)
        elif sample_type =='DDPM':
            
            x = self.ddpm_steps(x_T, z_sem, seq)
               
        return x[0][-1]
    
    
    def ddim_reverse_sample(
                            self,
                            x,
                            t,
                            z_sem,
        
                            clip_denoised=True,
                            denoised_fn=None,
                            model_kwargs=None,
                            eta=0.0,
                        ):
        
        assert eta == 0.0, "Reverse ODE only for deterministic path"
    
    
        
        out = self.p_mean_variance(
            x,
            t,
            z_sem,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape)
           * x - out["pred_xstart"]) / _extract_into_tensor(
               self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (out["pred_xstart"] * torch.sqrt(alpha_bar_next) +
                     torch.sqrt(1 - alpha_bar_next) * eps)

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}
    

    def stochastic_encode(self, x_0, T, z_sem):
        
        seq = self.get_timestep_sequence()
        device = x_0.device
        
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(self.T+1))
        
        sample = x_0
        
        for i in indices:
            t = torch.tensor([i] * len(sample), device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(
                                               sample,
                                               t=t,
                                               z_sem=z_sem
                                               )
                
                sample = out['sample']
                # [1, ..., T]
                sample_t.append(sample)
                # [0, ...., T-1]
                xstart_t.append(out['pred_xstart'])
                # [0, ..., T-1] ready to use
                T.append(t)
                
        return {
            #  xT "
            'sample': sample,
            # (1, ..., T)
            'sample_t': sample_t,
            # xstart here is a bit different from sampling from T = T-1 to T = 0
            # may not be exact
            'xstart_t': xstart_t,
            'T': T,
        }
    
    
    
    def trainer(self,
              adata, 
              output_model_file, 
              output_loss_file, 
              lr=1e-3):
        
    
        ann_data_dataset = datasets.AnnDataDataset(adata)
        dataloader = DataLoader(ann_data_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers, 
                                pin_memory=True)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)


        self.train()
        
        loss_all = []

        for epoch in range(self.n_train_epochs):
            epoch_loss = 0
            for data,features in dataloader:

                data = data.to(self.device)
                
                # antithetic sampling
                t_half = torch.randint(low=0, high=self.T+1, 
                                         size =(256, data.shape[0] // 2 + 1,)
                                        ).to(self.device)
                
                
                t_tensor = torch.cat([t_half, self.T - t_half ], dim=1)[:,:data.shape[0]]
                
                
                if len(range(torch.cuda.device_count()))>1:
                    loss, z_sem, x_t = self.module.p_loss(data, t_tensor)
                else:
                    loss, z_sem, x_t = self.p_loss(data, t_tensor)
                    
                
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()

                epoch_loss += loss.item()
                

            #scheduler.step()
            loss_all.append(epoch_loss / len(dataloader))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.n_train_epochs}; Loss: {epoch_loss / len(dataloader):.4f}")
                # print_gpu_utilization()  

            
        if len(range(torch.cuda.device_count()))>1:
            torch.save(self.module.state_dict(), output_model_file)
        else:
            torch.save(self.state_dict(), output_model_file)
        print(f"Model saved to {output_model_file}")

        with open(output_loss_file, 'w') as f:
            json.dump(loss_all, f)
        print(f"Loss data saved to {output_loss_file}")




def _extract_into_tensor(tensor, t, shape):
    extracted = tensor.gather(0, t).unsqueeze(-1)
    while len(extracted.shape) < len(shape):
        extracted = extracted.unsqueeze(-1)
    return extracted.expand(shape)
