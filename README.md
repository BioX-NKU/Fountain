[![PyPI](https://img.shields.io/pypi/v/scFountain.svg)](https://pypi.org/project/scFountain/)

# Rigorous integration of single-cell ATAC-seq data using regularized barycentric mapping
![](Fountain.png)

## Package installation

It is prefered to create a new environment for Fountain.

```
conda create -n Fountain python==3.8
conda activate Fountain
```

Fountain is available on [PyPI](https://pypi.org/project/scFountain/), and could be installed using

```
pip install scFountain
```

Installation via Github is also provided



This process will take approximately 2 to 10 minutes, depending on the user's computer device and internet connectivition.

## Tutorial

Usage and examples of Fountain's main functions are shown in [tutorial](https://github.com/BioX-NKU/Fountain/tree/main/Tutorials).



## Quick Start

Fountain is a deep learning framework for batch integration on scATAC-seq data utilizing  regularized barycentric mapping. Fountain can be easily used for: generating batch-corrected low-dimensional embeddings, generating batch-corrected and enhanced ATAC profiles in the original dimension, and online integration. 


### Input format
* **h5ad file**:
	* AnnData object of shape `n_obs` × `n_vars`. 
    
* **count matrix file**:  
	* Rows correspond to peaks and columns to cells.

* **batch label and cell type label**:  
	* The batch label and cell type labels are included in anndata.obs. Cell type labels are used for evaluation, rather than being necessary for training.


### 1. Data preprocessing

  ```python
import scanpy as sc
import episcanpy as epi
import numpy as np
import sklearn
import pandas as pd 
import torch
from Fountain.data import create_dataloader,create_batchind_dict
from Fountain.fountain import Fountain
import scib
import matplotlib.pyplot as plt
  ```
*  You can chick  [MB.h5ad](https://drive.google.com/file/d/1qwKP1xzYVs5rEGRJPU_NJga2Gl0qSTv5/view?usp=sharing) to download the example dataset. 



* After data preprocessing, you should load the raw count matrix scATAC-seq data via:
  
  ```python
  adata=sc.read("./MB.h5ad")
  fpeak=0.04
  epi.pp.binarize(adata)
  epi.pp.filter_features(adata, min_cells=np.ceil(fpeak*adata.shape[0]))
  ```
  
  
  Anndata object is a Python object/container designed for storing single-cell data in Python packege [**anndata**](https://anndata.readthedocs.io/en/latest/) which is seamlessly integrated with [**scanpy**](https://scanpy.readthedocs.io/en/stable/), a widely-used Python library for single-cell data analysis.

 
### 2. Model training

* Model initialization.

  
  ```python
  batchind_dict=create_batchind_dict(adata,batch_name='batch')
  batchsize=min(128*len(batchind_dict),1024)
  dataloader=create_dataloader(adata,batch_size=batchsize,batchind_dict=batchind_dict,batch_name='batch',num_worker=4,droplast=False)
  enc=[['fc', 1024, '', 'gelu'],['fc', 256, '', 'gelu'],['fc', 16, '', '']]
  dec=[['fc', adata.X.shape[1], '', '']]
  early_stopping= None
  device='cuda:0'
  ```



* Fountain model can be easily trained as following:
  
  ```python
  model.train(            
            dataloader,             
            lambda_mse=0.005, 
            lambda_Eigenvalue=0.5,
            max_iteration=30000,
            mid_iteration=3000,
            early_stopping=early_stopping,
            device=device, 
        )
  ```
  
  
### 3. Generating batch-corrected low-dimensional embeddings

* Fountain provides an API to get batch-corrected low-dimensional embeddings of scATAC-seq data. You can get batch-corrected embeddings by:
  
  ```python
  emb='fountain'
  adata.obsm[emb]=model.get_latent(dataloader,device=device)
  ```
* We provide codes to visualization the low-dimensional embeddings of data:

  ```python
  sc.pp.neighbors(adata, use_rep='fountain')
  sc.tl.umap(adata)
  sc.pl.umap(adata, color=['cell_type','batch'])
  ```

### 4. Generating batch-corrected and enhanced ATAC profiles in the original dimension

* Fountain provides an API to get batch-corrected and enhanced ATAC profiles in the original dimension. You can get the enhanced data by:
  
  ```python
  adata.layers['enhance']=model.enhance(adata,device=device,batch_name='batch')
  ```

### 5. Achieving online integration

* You can achieve online integration through the model.get_latent function. Please refer to the tutorial for more details.
  



