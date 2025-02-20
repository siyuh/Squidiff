import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import scanpy as sc
import os

class AnnDataDataset(Dataset):
    def __init__(self, adata):
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
            self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        self.obs = adata.obs.copy()
        
        
        self.encoded_obs = self.obs
        
    
        self.encoded_obs_tensor = torch.tensor(self.encoded_obs.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
       
        return self.features[idx], self.encoded_obs_tensor[idx]
    
def load_simulate_data(
        data_dir = None,
    ):
    train_adata = sc.read_h5ad(os.path.join(data_dir,'adata_simulate.h5ad'))
    test_adata  = sc.read_h5ad(os.path.join(data_dir,'adata_simulate.h5ad'))
    
    return train_adata, None, test_adata

    
def load_gene_pert_data(
        data_dir = None,
    ):
    train_adata = sc.read_h5ad(os.path.join(data_dir,'gears_train_data.h5ad'))
    test_adata  = sc.read_h5ad(os.path.join(data_dir,'gears_test_data.h5ad'))
    
    return train_adata, None, test_adata

def load_VO_data(
        data_dir = None,
    ):
    
    train_adata = sc.read_h5ad(os.path.join(data_dir,'VO_trained_adata.h5ad'))

    
    return train_adata, None, None

def load_VO_rad_data(
        data_dir = None,
    ):
    
    #train_adata = sc.read_h5ad(os.path.join(data_dir,'VO_rad_trained_adata.h5ad'))
    train_adata = sc.read_h5ad(os.path.join(data_dir,'VO_rad_trained_adata_wdiff.h5ad'))

    
    return train_adata, None, None

def load_VO_rad_drug_data(
        data_dir = None,
    ):
    
    #train_adata = sc.read_h5ad(os.path.join(data_dir,'VO_rad_trained_adata.h5ad'))
    train_adata = sc.read_h5ad(os.path.join(data_dir,'VO_rad_drug_trained_adata.h5ad'))

    
    return train_adata, None, None



def load_diff_data(
        data_dir = None,
    ):
    train_adata = sc.read_h5ad(os.path.join(data_dir,'train_adata.h5ad'))
    val_adata   = sc.read_h5ad(os.path.join(data_dir,'val_adata.h5ad'))
    test_adata  = sc.read_h5ad(os.path.join(data_dir,'test_adata.h5ad'))
    

    train_adata.obs['train_data'] = 1
    test_adata.obs['train_data'] = 0
    train_adata = train_adata[train_adata.obs['day'].isin(['day0','day3'])]

    combined_adata = sc.concat([train_adata,test_adata])
    sc.tl.pca(combined_adata, svd_solver='arpack')
    sc.pp.neighbors(combined_adata, n_neighbors=30, n_pcs=50)
    sc.tl.umap(combined_adata,min_dist=0.6)


    train_adata.obs = train_adata.obs[['day']]

    train_adata.obs['day'] = train_adata.obs['day'].map({
    'day0':0,
    'day1':1,
    'day2':2,
    'day3':3                                    
                                 })

    test_adata.obs = test_adata.obs[['day']]

    test_adata.obs['day'] = test_adata.obs['day'].map(
        {'day0':0,
         'day1':1,
         'day2':2,
         'day3':3,
    })

    val_adata.obs  = val_adata.obs[['day']]
    val_adata.obs['day'] = val_adata.obs['day'].map(
        {'day0':0,
         'day1':1,
         'day2':2,
         'day3':3,
    })

    
    return train_adata, val_adata, test_adata

    
def prepared_data(data_dir=None,
                  fn = None,
                 ):
     
    train_adata, val_adata, test_adata = fn(data_dir=data_dir)
    _data_dataset = AnnDataDataset(train_adata)
    
    dataloader = DataLoader(
                _data_dataset, 
                batch_size=64,
                shuffle=True, 
                )
    return dataloader