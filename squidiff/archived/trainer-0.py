
import torch
import torch.nn as nn
from . import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import json
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0


    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1


    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

class trainloop:
    def __init__(
        self,
        model, 
        diffusion,
        adata,
        model_kwargs=None,
        output_loss_file=None,
        output_model_file=None,
        output_ema_file=None,
        output_optim_file=None,
        log_file=None,
        train_bool=True,
        lr=1e-4,
        batch_size = 128,
        num_workers = 1,
        train_epochs = 1000,
        time_steps = 1000,
                ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.diffusion = diffusion.to(self.device)
        _data_dataset = datasets.AnnDataDataset(adata)

        self.dataloader = DataLoader(
            _data_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True)
        
        self.train_bool = train_bool
        self.scaler = torch.cuda.amp.GradScaler()
        self.ema = EMA(0.995)
        self.ema_model = self.model.__class__(**model_kwargs).to(self.device)
        self.ema.reset_parameters(self.ema_model, self.model)
        
        model_size = 0
        for param in self.model.parameters():
            model_size += param.data.nelement()
        print('Model params: %.2f M' % (model_size / 1024 / 1024))
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.001)
        self.loss_fn = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=lr, steps_per_epoch=len(self.dataloader), epochs=train_epochs)

        self.time_steps = time_steps
        self.train_epochs = train_epochs
        self.output_loss_file=output_loss_file
        self.output_model_file=output_model_file
        self.output_ema_file=output_ema_file
        self.output_optim_file=output_optim_file
        self.log_file = log_file
        
    def train(
        self,
    ):
        if self.train_bool: 
            self.model.train()
        else: 
            self.model.eval()
            
        loss_history = []
        log_data = []
        
        for epoch in range(self.train_epochs):

            epoch_loss = 0
            
            start_time = time.time()
            

            for data, features in self.dataloader:

                data = data.to(self.device)
                features = features.to(self.device)
                t_half = torch.randint(low=0, high=self.time_steps, size =(512, data.shape[0] // 2 + 1,))
                t_tensor = torch.cat([t_half, self.time_steps - t_half -1], dim=1)[:,:data.shape[0]]

                t_tensor = t_tensor.to(self.device)
                data = data.unsqueeze(0).expand(t_tensor.size(0), -1, -1).reshape(-1, data.size(-1))
                features = features.unsqueeze(0).expand(t_tensor.size(0), -1, -1).reshape(-1, features.size(-1))
                t_tensor = t_tensor.reshape(-1)

                noise = torch.randn_like(data).to(self.device)
                x_t = self.diffusion.q_sample(data, t_tensor, noise)
                z_sem = self.model.semantic_encoder(data).to(self.device)
                pred_noise = self.model(x_t, t_tensor, z_sem = z_sem, y=features[:,0])

                loss = self.loss_fn(pred_noise, noise)
                epoch_loss += loss.item()

                if self.train_bool:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.ema.step_ema(self.ema_model, self.model)
                    self.scheduler.step()
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            loss_history.append(avg_epoch_loss)
            end_time = time.time()
            epoch_time = end_time - start_time
            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            log_data.append(log_entry)
            
            if epoch%20==0:
                print(f"Epoch [{epoch+1}/{self.train_epochs}], Loss: {avg_epoch_loss:.4f}")

        print("Training complete.")


        # Save the loss history
        with open(self.output_loss_file, 'w') as f:
            json.dump(loss_history, f)

        torch.save(self.model.state_dict(), self.output_model_file)
        torch.save(self.ema_model.state_dict(), self.output_ema_file)
        torch.save(self.optimizer.state_dict(), self.output_optim_file)

        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f)


