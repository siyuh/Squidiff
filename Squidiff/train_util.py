"""
This code is adapted from openai's guided-diffusion models and Konpat's diffae model:
https://github.com/openai/guided-diffusion
https://github.com/phizaz/diffae
"""
import copy
import functools
import os

import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import matplotlib.pyplot as plt
import numpy as np

INITIAL_LOG_LOSS_SCALE = 20.0

def plot_loss(losses,args_train):
    # Convert losses to a numpy array
    losses_np = np.array([i.detach().cpu().numpy() for i in losses])

    # Define the window size for the moving average
    window_size = int(args_train['lr_anneal_steps']/1000)+1  # You can adjust this value based on your preference

    # Calculate the moving average (mean of the windowed losses)
    windowed_mean_loss = np.convolve(losses_np, np.ones(window_size) / window_size, mode='valid')

    # Adjust the x-axis values for the windowed mean loss
    x_vals = np.linspace(0, args_train['lr_anneal_steps']-1, len(losses_np))
    windowed_x_vals = x_vals[window_size - 1:]  # Adjust to match the length of the windowed_mean_loss

    # Plotting
    fig,ax = plt.subplots(figsize=(4.5, 3.4), dpi=800)
    plt.plot(x_vals, losses_np, label='Training Loss', alpha=0.2)
    plt.plot(windowed_x_vals, windowed_mean_loss, label='Windowed Mean Loss', color='r', linewidth=1)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    fig.savefig(args_train['resume_checkpoint']+'/loss_plot.pdf', transparent=True)

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        use_drug_structure=False,
        comb_num=1
    ):
        
        self.model = model
        self.diffusion = diffusion
        self.use_drug_structure = use_drug_structure
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size #* dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self.loss_list = []
        #self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            #self.ddp_model = DDP(
            #    self.model,
            #    device_ids=[dist_util.dev()],
            #    output_device=dist_util.dev(),
            #    broadcast_buffers=False,
            #    bucket_cap_mb=128,
            #    find_unused_parameters=False,
            #)
            self.ddp_model = self.model
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
            
        

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
           
            
            batch = next(iter(self.data))

            self.run_step(batch)
            
            
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch):
        self.forward_backward(batch)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        
        for i in range(0, batch['feature'].shape[0], self.microbatch):
            
            micro = batch['feature'][i : i + self.microbatch].to(dist_util.dev())
            if self.use_drug_structure:
                micro_cond = {
                    'group': batch['group'][i : i + self.microbatch],
                    'drug_dose': batch['drug_dose'][i : i + self.microbatch].to(dist_util.dev()),
                }
            else:
                micro_cond = {
                    'group': batch['group'][i : i + self.microbatch],
                    'drug_dose':None
                }
            
            last_batch = (i + self.microbatch) >= batch['feature'].shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            elif hasattr(self.ddp_model, 'no_sync'):
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            else:
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            
            self.mp_trainer.backward(loss)
            
        self.loss_list.append(loss)
        #print('loss=',loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params) if self.mp_trainer else self.model.state_dict()
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not os.path.exists(self.resume_checkpoint):
                    # Directory doesn't exist, so create it
                    os.makedirs(self.resume_checkpoint)
                if not rate: 
                    filepath = os.path.join(self.resume_checkpoint, "model.pt")
                else:
                    filepath = os.path.join(self.resume_checkpoint, f"model_{rate}.pt")
                with open(filepath, "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params if self.mp_trainer else self.model.parameters())
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            opt_filename = f"opt{(self.step+self.resume_step):06d}.pt"
            opt_filepath = os.path.join(get_blob_logdir(), opt_filename)
            with open(opt_filepath, "wb") as f:
                th.save(self.opt.state_dict(), f)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = os.path.join(os.path.dirname(main_checkpoint), filename)
    if os.path.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
