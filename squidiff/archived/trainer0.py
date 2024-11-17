
import torch
import torch.nn as nn
from . import datasets, resample
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import json
import umap
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context

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

def plot_intermediate(test_adata,noise_pred_model,diff_model):
    noise_pred_model.eval()
    
    data = torch.Tensor(test_adata.to_df().values).to('cuda')
    label = torch.Tensor(test_adata.obs.values).to('cuda')
    cond = noise_pred_model.encode(data,label)['cond']
    
    xT = diff_model.encode_stochastic(noise_pred_model, 
                                      data, 
                                      cond,
                                      label
                                     )



    reducer = umap.UMAP(n_neighbors=30, min_dist=0.02, n_components=2)
    embedding = reducer.fit_transform(cond.detach().cpu())

    plt.figure(figsize=(6,4),dpi=600)
    colorlist =  ['#3145a8', '#40a8f7', '#f5bf36','#fa2616']
    for i in range(4):
        scatter = plt.scatter(embedding[test_adata.obs['day'].values==i, 0], 
                          embedding[test_adata.obs['day'].values==i, 1], 
                          c= colorlist[i],
                          s=5,
                          alpha=1)
    #plt.colorbar(scatter, label='Day')
    #plt.title('UMAP projection of z_sem embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2'
              )
    plt.axis('off')
    plt.show()

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.02, n_components=2)
    embedding = reducer.fit_transform(xT['sample'].detach().cpu())

    plt.figure(figsize=(6,4),dpi=600)
    colorlist =  ['#3145a8', '#40a8f7', '#f5bf36','#fa2616']
    for i in range(4):
        scatter = plt.scatter(embedding[test_adata.obs['day'].values==i, 0], 
                          embedding[test_adata.obs['day'].values==i, 1], 
                          c= colorlist[i],
                          s=5,
                          alpha=1)
    #plt.colorbar(scatter, label='Day')
    #plt.title('UMAP projection of z_sem embeddings')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2'
              )
    plt.axis('off')
    plt.show()
    
class trainloop:
    def __init__(
        self,
        model, 
        diffusion,
        adata,
        val_data,
        model_kwargs=None,
        output_loss_file=None,
        output_model_file=None,
        output_ema_file=None,
        output_optim_file=None,
        log_file=None,
        train_bool=True,
        lr=1e-3,
        batch_size = 128,
        num_workers = 1,
        train_epochs = 1000,
        time_steps = 1000,
                ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_steps = time_steps
        self.model = model.to(self.device)
        self.diffusion = diffusion.to(self.device)
        self.T_sampler = resample.UniformSampler(self.time_steps)
        
        
        
        _data_dataset = datasets.AnnDataDataset(adata)
        if val_data is not None:
            _val_data_dataset = datasets.AnnDataDataset(val_data)
            self.val_dataloader = DataLoader(
            _val_data_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True)

        self.dataloader = DataLoader(
            _data_dataset, 
            batch_size=batch_size,
           shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True)
        
        
        self.train_bool = train_bool
        self.scaler = torch.cuda.amp.GradScaler()
        
        model_size = sum(param.data.nelement() for param in self.model.parameters())
        print('Model params: %.2f M' % (model_size / 1024 / 1024))
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.001)
        
        #self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #    self.optimizer, 
        #    max_lr=lr, 
        #    steps_per_epoch=len(self.dataloader), 
        #    epochs=train_epochs
        #)

        self.train_epochs = train_epochs
        self.output_loss_file = output_loss_file
        self.output_model_file = output_model_file
        self.output_ema_file = output_ema_file
        self.output_optim_file = output_optim_file
        self.log_file = log_file
        
        
    def train(self, test_adata=None,):
        
        self.val_loss_history = []
        self.loss_history = []
        log_data = []
        val_loss_history = []
        self.log_noise_min = []
        self.log_noise_max = []
        self.log_noise_ori_min = []
        self.log_noise_ori_max = []
        for epoch in range(self.train_epochs):
            
            self.model.train() if self.train_bool else self.model.eval()
            epoch_loss = 0
            start_time = time.time()
            
            for x0, features in self.dataloader:
                

                x0 = x0.to(self.device)
                features = features.to(self.device)
                t, weight = self.T_sampler.sample(len(x0), x0.device)
                x0 = x0.repeat(t.shape[0] // x0.shape[0], 1,1).reshape([t.shape[0],-1])

                losses = self.diffusion.training_losses(model=self.model, x_start=x0, t=t, label = features)
                
                
                
                loss = losses['loss'].mean()
                epoch_loss += loss.item()
                if self.train_bool:
                    self.optimizer.zero_grad()  
                    
                    self.optimizer.step()  
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    #self.scheduler.step()
                    # self.ema.step_ema(self.ema_model, self.model)
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            self.loss_history.append(avg_epoch_loss)
            end_time = time.time()
            epoch_time = end_time - start_time
            log_entry = {
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "time": epoch_time,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            log_data.append(log_entry)
            
            if epoch%100==0:
                print(f"Epoch [{epoch+1}/{self.train_epochs}], Loss: {avg_epoch_loss:.4f}")
                #plot_intermediate(test_adata,self.model,self.diffusion)
            
            #if self.val_dataloader is not None and epoch % 20 == 0:
            if 0>1:
                self.model.eval()
                val_dataloader = DataLoader(
                    datasets.AnnDataDataset(test_adata), 
                    batch_size=self.dataloader.batch_size, 
                    shuffle=False, 
                    num_workers=self.dataloader.num_workers, 
                    pin_memory=True
                )
                val_epoch_loss = 0
                with torch.no_grad():
                    for x0, features in val_dataloader:
                        x0 = x0.to(self.device)
                        features = features.to(self.device)
                        t, weight = self.T_sampler.sample(len(x0), x0.device)
                        x0 = x0.repeat(t.shape[0] // x0.shape[0], 1, 1).reshape([t.shape[0], -1])

                        losses = self.diffusion.training_losses(model=self.model, x_start=x0, t=t)
                        loss = losses['loss'].mean()
                        val_epoch_loss += loss.item()

                avg_val_epoch_loss = val_epoch_loss / len(val_dataloader)
                self.val_loss_history .append(avg_val_epoch_loss)
                print(f"Validation Loss after Epoch [{epoch + 1}/{self.train_epochs}]: {avg_val_epoch_loss:.4f}")
 
        print("Training complete.")

        self.loss_history = [float(loss) for loss in self.loss_history]
        
        torch.save(self.model.state_dict(), self.output_model_file)
        # torch.save(self.ema_model.state_dict(), self.output_ema_file)
        torch.save(self.optimizer.state_dict(), self.output_optim_file)

        with open(self.output_loss_file, 'w') as f:
            json.dump(self.loss_history, f)
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f)


