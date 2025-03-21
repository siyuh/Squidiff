# -*- coding: utf-8 -*-
# @Author: SiyuHe
# @Last Modified by:   SiyuHe
# @Last Modified time: 2025-02-19


import argparse
import os
import numpy as np
import torch.distributed as dist
import torch
from Squidiff import dist_util, logger
from Squidiff.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from Squidiff.resample import create_named_schedule_sampler
from sklearn.metrics import r2_score
import scipy
class sampler:
    def __init__(self,model_path,gene_size,output_dim,use_drug_structure):
        args = self.parse_args(model_path,gene_size,output_dim,use_drug_structure)
        print("load model and diffusion...")

        model, diffusion = create_model_and_diffusion(
                **args_to_dict(args, model_and_diffusion_defaults().keys())
            )

        model.load_state_dict(
            dist_util.load_state_dict(args['model_path'])
        )
        model.to(dist_util.dev())
        if args['use_fp16']:
            model.convert_to_fp16()
        model.eval()
        self.model = model
        self.arg = args
        self.diffusion = diffusion
        self.sample_fn = (diffusion.p_sample_loop if not args['use_ddim'] else diffusion.ddim_sample_loop)
    
    def stochastic_encode(
        self, model, x, t, model_kwargs):
        """
        ddim reverse sample
        """
        sample = x
        sample_t = []
        xstart_t = []
        T = []
        indices = list(range(t))

        for i in indices:
            timestep = torch.full((x.shape[0],), i, device='cuda').long()
            with torch.no_grad():
                out = self.diffusion.ddim_reverse_sample(model, 
                                                    sample, 
                                                    timestep, 
                                                    model_kwargs=model_kwargs)
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

    def parse_args(self,model_path,gene_size,output_dim,use_drug_structure):
        """Parse command-line arguments and update with default values."""
        # Define default arguments
        default_args = {}
        default_args.update(model_and_diffusion_defaults())
        updated_args = {
            'data_path': '',
            'schedule_sampler': 'uniform',
            'lr': 1e-4,
            'weight_decay': 0.0,
            'lr_anneal_steps': 1e5,
            'batch_size': 16,
            'microbatch': -1,
            'ema_rate': '0.9999',
            'log_interval': 1e4,
            'save_interval': 1e4,
            'resume_checkpoint': '',
            'use_fp16': False,
            'fp16_scale_growth': 1e-3,
            'gene_size': gene_size,
            'output_dim': output_dim,
            'num_layers': 3,
            'class_cond': False,
            'use_encoder': True,
            'use_ddim':True,
            'diffusion_steps': 1000,
            'logger_path': '',
            'model_path': model_path,
            'use_drug_structure':use_drug_structure,
            'comb_num':1,
            'drug_dimension':1024
        }
        default_args.update(updated_args)

        # Return the updated arguments as a dictionary
        return default_args

    def load_squidiff_model(self):
        print("load model and diffusion...")
        return self.model

    def load_sample_fn(self):
        
        return self.sample_fn

    def get_diffused_data(self,model, x, t, model_kwargs):
        sample = x
        sample_t = [x]  # Store initial data for plotting
        xstart_t = []
        T = []

        indices = list(range(t))

        for i in indices:
            timestep = torch.full((x.shape[0],), i, device='cuda').long()
            with torch.no_grad():
                # Replacing ddim_reverse_sample with a simpler forward diffusion process
                noise = torch.randn_like(sample)  # Add noise at each step
                out = sample + noise * (i / t)    # Simulating diffusion based on time step
                sample = out
                sample_t.append(sample.cpu())  # Store the samples for visualization
                xstart_t.append(sample.cpu())
                T.append(timestep)

        return {
            'sample': sample,
            'sample_t': sample_t,
            'xstart_t': xstart_t,
            'T': T
        }

    def sample_around_point(self, point, num_samples=None, scale=0.7):
        return point + scale * np.random.randn(num_samples, point.shape[0])

    def pred(self,z_sem, gene_size):
        
        pred_result = self.sample_fn(
                        self.model,
                        shape = (z_sem.shape[0], gene_size),
                        model_kwargs={
                            'z_mod': z_sem
                        },
                        noise =  None
                )
        return pred_result
    
    def interp_with_direction(self, z_sem_origin = None, gene_size = None, direction = None, scale = 1, add_noise_term = True):

        z_sem_origin = z_sem_origin.detach().cpu().numpy()
        z_sem_interp_ = z_sem_origin.mean(axis=0) + direction.detach().cpu().numpy() * scale
        if add_noise_term:
            z_sem_interp_ = self.sample_around_point(z_sem_interp_, num_samples=z_sem_origin.shape[0])

        z_sem_interp_ = torch.tensor(z_sem_interp_,dtype=torch.float32).to('cuda')
        sample_interp = self.sample_fn(
                            self.model,
                            shape = (z_sem_origin.shape[0],gene_size),
                            model_kwargs={
                                'z_mod': z_sem_interp_
                            },
                            noise =  None
        )
        return sample_interp
        
    def cal_metric(self,x1,x2):
        r2 = r2_score(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        pearsonr,_ = scipy.stats.pearsonr(x1.detach().cpu().numpy().mean(axis=0),
                      x2.X.mean(axis=0))
        return r2, pearsonr

        

