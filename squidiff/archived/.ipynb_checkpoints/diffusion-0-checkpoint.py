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
from . import datasets

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)

class SquidModel(nn.Module):
    def __init__(
        self, 
         input_dim, 
         z_sem_dim, 
         time_steps=None, 
         time_emb_dim=80, 
         hidden_dim=512, 
         batch_size=16, 
         num_workers=1, 
         model_var_type='fixed_large',
                 reverse_steps=100, 
                 rescale_timesteps=False):
        super().__init__()
        
        self.num_timesteps = time_steps
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_var_type = model_var_type
        self.reverse_steps = reverse_steps
        self.num_workers = num_workers
        self.rescale_timesteps = rescale_timesteps
    

        self.betas = torch.linspace(0.0001, 0.02, self.num_timesteps).to(self.device)
        
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
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

        
    def q_mean_variance(
        self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        """
        mean = (_extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_start.shape)
        return mean, variance, log_variance

    
    def q_sample(
        self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        sample from q(x_t | x_0).
        """
        
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
    
    def q_posterior_mean_variance(
        self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (_extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                          _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] == x_start.shape[0])
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    
    def ddim_sample(
        self,
        model,
        x,
        z_sem,
        t,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            z_sem,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    def ddim_sample_loop(
        self,
        model,
        z_sem,
        x_T,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        """
        
        indices = list(range(self.num_timesteps))[::-1]
        x = x_T
        samples = []
        for i in indices:
            t = torch.tensor([i] * x.shape[0], device=self.device)
            with torch.no_grad():
                out = self.ddim_sample(
                    model,
                    x,
                    z_sem,
                    t,
                )
                samples.append(out)
                x = out["sample"]
        return x


    def p_mean_variance(
        self, 
        model, 
        x, 
        t, 
        z_sem, 
        model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
        
        B, C = x.shape[:2]
        assert t.shape == (B,)
        t = self._scale_timesteps(t)
        
        #print(x.shape)
        #print(t.shape)
        #print(z_sem.shape)
        model_output = model(x, t, z_sem=z_sem, **model_kwargs)

        if self.model_var_type in ['learned', 'learned_range']:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = torch.split(model_output, C, dim=1)
            if self.model_var_type == 'learned':
                model_log_variance = model_var_values
                model_variance = torch.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(torch.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = torch.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                'fixed_large': (
                    torch.cat([self.posterior_variance[1:2], self.betas[1:]]),
                    torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])),
                ),
                'fixed_small': (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            
            return x

        pred_xstart = process_xstart(self._predict_xstart_from_eps(x, t, model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return {"mean": model_mean, "variance": model_variance, "log_variance": model_log_variance, "pred_xstart": pred_xstart}

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))


    
    def _predict_xstart_from_eps(
        self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def _scale_timesteps(
        self, t):
        return t.float() * (1000.0 / self.num_timesteps) if self.rescale_timesteps else t
    

    def ddim_reverse_sample(self, model, x, t, z_sem):
        out = self.p_mean_variance(model, x, t, z_sem, model_kwargs={})
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps
        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}
    
    def stochastic_encode(
        self, model, x_t, z_sem, t):
        """
        ddim reverse sample
        """
        sample = x_t
        sample_t = []
        xstart_t = []
        T = []
        device = x_t.device
        indices = list(range(t.unique(), self.num_timesteps))

        for i in indices:
            timestep = torch.tensor([i] * len(sample), device=device)
            with torch.no_grad():
                out = self.ddim_reverse_sample(model, sample, timestep, z_sem)
                sample = out['sample']
                sample_t.append(sample)
                xstart_t.append(out['pred_xstart'])
                T.append(timestep)

        return {
        'sample': sample,
        'sample_t': sample_t,
        'xstart_t': xstart_t,
        'T': T,
    }
    
    def reconstruct(self,pred_model, z_sem, x_T, eta = 0 ):
        """
        Reconstruct the original data x_0 from noisy data x_T using reverse diffusion.
        """
        #x = x_T.float()
        seq = list(range(self.time_steps))
        with torch.no_grad(): 
            n = x_T.size(0)
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs=[x_T]
            
            for i, j in zip(reversed(seq),reversed(seq_next)):
                if j>-1:
                    t = torch.full((n,), i, dtype=torch.long, device=self.device)
                    next_t = torch.full((n,), j, dtype=torch.long, device=self.device)

                    at = self.compute_alpha(t)
                    at_next = self.compute_alpha(next_t)

                    xt = xs[-1]

                    epsilon_theta = pred_model(xt, t)
                    x0_t = (xt - epsilon_theta * torch.sqrt(1 - at)) / torch.sqrt(at)
                    x0_preds.append(x0_t)

                    xt_next = x0_t * torch.sqrt(at_next) + epsilon_theta * torch.sqrt(1 - at_next)

                    xs.append(xt_next)
                
        return xs[-1], x0_preds
        
    
    def stochastic_encode_test(self, x_0, T, z_sem):
        device = x_0.device
        x_t = x_0

        for t in range(0, self.num_timesteps-1):
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
    


    


def _extract_into_tensor(
    tensor, t, shape):
    extracted = tensor.gather(0, t).unsqueeze(-1)
    while len(extracted.shape) < len(shape):
        extracted = extracted.unsqueeze(-1)
    return extracted.expand(shape)