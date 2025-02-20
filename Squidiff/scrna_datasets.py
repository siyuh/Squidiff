import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import scanpy as sc
import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    adopted from PRnet @Author: Xiaoning Qi.
    Encode SMILES of drug to rFCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    if comb_num==1:
        for i, smiles in enumerate(drug_SMILES_list):
            smi = smiles
            mol = Chem.MolFromSmiles(smi)
            fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
            fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
            fcfp4_list = fcfp4_list*np.log10(dose_list[i]+1)
            fcfp4_array[i] = fcfp4_list
    else:
        for i, smiles in enumerate(drug_SMILES_list):
            smiles_list = smiles.split('+')
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
                fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
                fcfp4_list = fcfp4_list*np.log10(float(dose_list[i])+1)
                fcfp4_array[i] += fcfp4_list
    return fcfp4_array 

class AnnDataDataset(Dataset):
    def __init__(self, adata,use_drug_structure,comb_num):
        self.use_drug_structure = use_drug_structure
        if type(adata.X)==np.ndarray:
            self.features = torch.tensor(adata.X, dtype=torch.float32)
        else:
            self.features = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        
        if self.use_drug_structure:
            self.drug_type_list = adata.obs['SMILES'].to_list()
            self.dose_list = adata.obs['dose'].to_list()
            #self.encoded_obs_tensor = torch.tensor(adata.obs['Group'].copy().values, dtype=torch.float32)
            self.encoded_obs_tensor = adata.obs['Group'].copy().values
            self.encode_drug_doses = Drug_dose_encoder(self.drug_type_list, self.dose_list, comb_num=comb_num)
            self.encode_drug_doses = torch.tensor(self.encode_drug_doses, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
       
        if self.use_drug_structure:
            return {'feature':self.features[idx], 'drug_dose':self.encode_drug_doses[idx], 'group': self.encoded_obs_tensor[idx]}
        else:
            return {'feature':self.features[idx], 'drug_dose':None, 'group': self.encoded_obs_tensor[idx]}
            
    

def prepared_data(data_dir=None,batch_size=64,use_drug_structure=False,comb_num=1):
     
    train_adata = sc.read_h5ad(data_dir)
    
    _data_dataset = AnnDataDataset(train_adata,use_drug_structure,comb_num)
    
    
    dataloader = DataLoader(
                _data_dataset, 
                batch_size=batch_size,
                shuffle=True, 
                )
    return dataloader