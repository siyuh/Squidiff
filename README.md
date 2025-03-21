<img src="squidiff_logo.png" width="80" /> **squidiff: Predicting cellular development and responses to perturbations using a diffusion model**
---
Squidiff is a diffusion model-based generative framework designed to predict transcriptomic changes across diverse cell types in response to a wide range of environmental changes.

<img src=squidiff_fig.png width="1000" />

### Installation
`pip install Squidiff`

### Model Input:
h5ad file with info: 
- Single-cell count matrix
- Meta data
- (optional) additional drug compounds

### Features 
- Predicting single-cell transcriptomics upon drug treatments 
- Predicting cell differentiation 
- Predicting gene perturbation

### Training Squidiff
```
python train_squidiff.py --logger_path LOGGER_FIRE_NAME --data_path YOUR_ADATASET.h5ad --resume_checkpoint ptNAME --gene_size 500 --output_dim 500
```
For incorporating drug structure in training, see the example: 
```
python train_squidiff.py --logger_path logger_files/logger_sciplex_random_split_0 --data_path datasets/sci_plex_train_random_split_0.h5ad --resume_checkpoint sciplex_results_random_split_0 --use_drug_structure True --gene_size 200 --output_dim 200 --control_data_path datasets/sci_plex_train_random_split_0_control.h5ad
```
### Sample Squidiff
```python
sampler = sample_squidiff.sampler(
    model_path = 'simu_results/model.pt',
    gene_size = 100,
    output_dim = 100,
    use_drug_structure = False
)

test_adata_scrna = sc.read_h5ad('datasets/sc_simu_test.h5ad')
z_sem_scrna = sampler.model.encoder(torch.tensor(test_adata_scrna.X).to('cuda'))

scrnas_pred = sampler.pred(z_sem_scrna, gene_size = test_adata_scrna.shape[1])


```
### How to cite Squidiff

bioRxiv: https://doi.org/10.1101/2024.11.16.623974

## Contact
In case you have questions, please contact:
- Siyu He - siyuhe@stanford.edu
- via Github Issues
