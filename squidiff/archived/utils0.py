import sys
import pandas as pd
import scanpy as sc
import numpy as np
import torch
import os
import anndata
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib.pyplot import rc_context
import umap
import time
import phenograph
from scipy.stats import zscore
import torch.nn as nn
import math


def timestep_embedding(time, dim=128):
    """
    Compute sinusoidal position embeddings.

    :param time: Input tensor of timesteps.
    :param dim: Dimension of the embeddings.
    :return: Sinusoidal position embeddings.
    """
    device = time.device
    half_dim = dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings



class sc_object:
    """
    adata_raw: raw counts only with pre-identified information
    adata_normed: normalized and logirimized version
    adata_counts: raw countds with filtered genes and cells
    """
    def load_sc_data(self,data_paths,groups):
        """
        if start with file paths, no need to run set_up again
        """
        self.adata_raw = []
        for i in data_paths:
            assert os.path.exists(i), f"file didn't find with this path: {i}" 
            if i.endswith('h5ad'):  
                adata_ = sc.read_h5ad(i)
            elif i.endswith('h5'):
                adata_ = sc.read_10x_h5(i)
            adata_.var_names_make_unique()
            adata_.obs_names = adata_.obs_names+'_'+str(groups[data_paths.index(i)])
            adata_.obs['group']=groups[data_paths.index(i)]
            adata_.obs['group'] = adata_.obs['group'].astype('category')
            self.adata_raw.append(adata_)
        self.adata_raw = anndata.concat(self.adata_raw)
        pass
    
    
    def set_up(self,adata):
        """
        if start with anndata, run set_up first
        """
        adata_=adata.copy()
        #adata_.var_names_make_unique()
        #adata_.obs_names = adata_.obs_names+'_'+str(group)
        #adata_.obs['group']=group
        #adata_.obs['group'] = adata_.obs['group'].astype('category')
        self.adata_raw=adata_
        pass
        
    
    def preprocess(self,filter_ = True, n_top_genes=2000): 
        
        print('preprocessing adata')
        adata = self.adata_raw.copy()
        adata.var['mt'] = adata.var_names.str.startswith('MT-')
        adata.var['rb'] = np.logical_or(
        adata.var_names.str.startswith('RPS'),
        adata.var_names.str.startswith('RPL')
    )
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
        
        if filter_:
            min_perc=0.005
            min_counts = np.percentile(adata.obs['total_counts'], min_perc)
            print('filtering cells... min_genes ='+ str(min_counts))
            sc.pp.filter_cells(adata, min_genes=min_counts)

            print('filtering genes... min_cells = 3')
            sc.pp.filter_genes(adata, min_cells=3)

            print('filtering cells... based on n_genes & pct_mt')
            adata = adata[adata.obs.n_genes_by_counts < 1e4, :]
            adata = adata[adata.obs.pct_counts_mt < 20, :]
            adata = adata[adata.obs.n_genes_by_counts > 1000, :]
            mask_gene = np.logical_and(~adata.var['mt'], ~adata.var['rb'])

            print('filtering out... mt & rp genes')
            adata = adata[:, mask_gene]
        
        print('normalizing data')
        sc.pp.normalize_total(adata, target_sum=1e6)
        print('logarithm data')
        sc.pp.log1p(adata)
        
        print('finding high variable genes')
        print(adata)
        sc.pp.highly_variable_genes(adata,n_top_genes=n_top_genes, inplace=True)
        
        self.adata_normed = adata
        self.highly_variable_genes = adata.var['highly_variable']
        self.adata_counts = self.adata_raw.copy()[adata.obs_names,adata.var_names]
        
        
    def get_umap(self):
        sc.tl.pca(self.adata_normed, n_comps=100)
        sc.pp.neighbors(self.adata_normed, n_neighbors=30, n_pcs=50)
        self.adata_normed.obsm['X_pca'] = self.adata_normed.obsm['X_pca']
        self.adata_counts.obsm['X_pca'] = self.adata_normed.obsm['X_pca']
        fit = umap.UMAP(
            n_neighbors=45,
            min_dist=0.5,
                   )
        t0 = time.time()
        u = fit.fit_transform(self.adata_normed.obsm['X_pca'])
        t1 = time.time()
        print('CPU times = ',t1-t0) 
        
        self.adata_normed.obs['umap_X'] = u[:,0]
        self.adata_normed.obs['umap_Y'] = u[:,1]
        
        self.adata_counts.obs['umap_X'] = u[:,0]
        self.adata_counts.obs['umap_Y'] = u[:,1]
        pass
    
    
    def get_cluster(self):
        print('getting clusters with phenograph')
        k = 30
        communities, graph, Q = phenograph.cluster(pd.DataFrame(self.adata_normed.obsm['X_pca']),k=k) # run PhenoGraph

        self.adata_normed.obs['PhenoGraph_clusters'] = pd.Categorical(communities)
        self.adata_normed.uns['PhenoGraph_Q'] = Q
        self.adata_normed.uns['PhenoGraph_k'] = k
        
        self.adata_counts.obs['PhenoGraph_clusters'] = pd.Categorical(communities)
        self.adata_counts.uns['PhenoGraph_Q'] = Q
        self.adata_counts.uns['PhenoGraph_k'] = k
         
    def get_adata(self, normed=False, high_var = False):
        if normed:
            if high_var:
                return self.adata_normed[:,self.highly_variable_genes]
            else:
                return self.adata_normed
        else:
            if high_var:
                return self.adata_counts[:,self.highly_variable_genes]
            else:
                return self.adata_counts
            
    def ct_prior(self, marker_list):
        adata = self.adata_normed
        for i in marker_list.columns:
            marker_ = np.intersect1d(list(marker_list[i]),adata.var_names)
            adata.obs[i]=list(adata[:,marker_].to_df().mean(axis=1))
        adata_prior = adata.obs[marker_list.columns]
        temp_ = adata_prior.apply(zscore,axis=1)
        temp_ = temp_.fillna(0)
        adata_prior = temp_ - temp_.min().min()+1e-5
        self.adata_prior=adata_prior
        pass
            
        
def model_eval(
    model,
    adata_sample,
    device,
):
    
    model.eval()
    
    group_ = torch.Tensor(np.array(adata_sample.obs['group_id']))
    x_valid = torch.Tensor(np.array(adata_sample.to_df()))
    x_valid = x_valid.to(device)
    
    library = torch.log(x_valid.sum(1)).unsqueeze(1)

    inference_outputs =  model.inference(x_valid, group_)
    generative_outputs = model.generative(inference_outputs)

    px = VAE5.NegBinom(generative_outputs["px_rate"], torch.exp(generative_outputs["px_r"])).sample().detach().cpu().numpy()
    return inference_outputs, generative_outputs, px




def plot_sinusoidal_embeddings(embedding_layer, num_steps=100):
    # Generate a range of time steps
    time_steps = torch.arange(num_steps).float()

    # Obtain embeddings
    embeddings = embedding_layer(time_steps).detach().numpy()

    # Plot
    plt.figure(figsize=(15, 6))
    for i in range(min(5, embeddings.shape[1])):  # Plot first 5 dimensions
        plt.plot(time_steps.numpy(), embeddings[:, i], label=f'Dim {i}')

    plt.xlabel('Time step')
    plt.ylabel('Sinusoidal embedding')
    plt.title('Sinusoidal Time Embeddings')
    plt.legend()
    plt.show()

    

def generate_umap(noisy_data, n_neighbors=15, min_dist=0.1, n_components=2, random_state=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    embedding = reducer.fit_transform(noisy_data)
    return embedding




from scipy.stats import gaussian_kde
def display_reconst(df_true,
                    df_pred,
                    density=False,
                    den_all = False,
                    marker_genes=None,
                    sample_rate=0.1,
                    size=(4, 4),
                    spot_size=1,
                    title=None,
                    x_label='',
                    y_label='',
                    ):
    """
    Scatter plot - raw gexp vs. reconstructed gexp
    """
    assert 0 < sample_rate <= 1, \
        "Invalid downsampling rate for reconstruct scatter plot: {}".format(sample_rate)

    if marker_genes is not None:
        marker_genes = set(marker_genes)

    df_true_sample = df_true.sample(frac=sample_rate, random_state=0)
    df_pred_sample = df_pred.loc[df_true_sample.index]

    plt.rcParams["figure.figsize"] = size
    plt.figure(dpi=800)
    ax = plt.gca()

    xx = df_true_sample.T.to_numpy().flatten()
    yy = df_pred_sample.T.to_numpy().flatten()

    if density and not den_all:
        for gene in df_true_sample.columns:
            try:
                gene_true = df_true_sample[gene].values
                gene_pred = df_pred_sample[gene].values
                gexp_stacked = np.vstack([df_true_sample[gene].values, df_pred_sample[gene].values])

                z = gaussian_kde(gexp_stacked)(gexp_stacked)
                ax.scatter(gene_true, gene_pred, c=z,vmin=0,cmap='Spectral_r',
                           s=spot_size, alpha=0.6)
            except np.linalg.LinAlgError as e:
                pass

    elif marker_genes is not None:
        color_dict = {True: 'red', False: 'green'}
        gene_colors = np.vectorize(
            lambda x: color_dict[x in marker_genes]
        )(df_true_sample.columns)
        colors = np.repeat(gene_colors, df_true_sample.shape[0])

        ax.scatter(xx, yy, c=colors, s=spot_size, alpha=0.5)

    elif density and den_all:
        gene_true = df_true_sample.values.flatten()
        gene_pred = df_pred_sample.values.flatten()
        gexp_stacked = np.vstack([df_true_sample.values.flatten(), df_pred_sample.values.flatten()])
        z = gaussian_kde(gexp_stacked)(gexp_stacked)
        ax.scatter(gene_true, gene_pred, c=z,vmin=0,cmap='Spectral_r',
                           s=spot_size, alpha=0.99)
    else:
        ax.scatter(xx, yy, s=spot_size, alpha=0.99)
    
    
        
    min_val = min(xx.min(), yy.min())
    max_val = max(xx.max(), yy.max())
    
    
    #ax.set_aspect('equal',adjustable='box')
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 14)
    #ax.axis('equal')
    
    plt.suptitle(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #ax.set_xlim(-0.5, 12)
    #ax.set_ylim(-0.5, 12)

    plt.show()
    
    
    
def slerp(start, direction, weight):
    """Spherical linear interpolation using a start tensor and a direction tensor"""
    end = start + direction
    
    start_norm = start / torch.norm(start, p=2, dim=-1, keepdim=True)
    end_norm = end / torch.norm(end, p=2, dim=-1, keepdim=True)

    dot_product = torch.sum(start_norm * end_norm, dim=-1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    omega = torch.acos(dot_product)
    sin_omega = torch.sin(omega)

    s1 = torch.sin((1.0 - weight) * omega) / (sin_omega)
    s2 = torch.sin(weight * omega) / (sin_omega)

    return s1 * start + s2 * end

def lerp(start, weight, direction):
    """Linear interpolation with a start tensor and a direction tensor"""
    return start + weight * direction