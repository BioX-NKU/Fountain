import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse, csr
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from anndata import AnnData
import scanpy as sc
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

#from glob import glob

#np.warnings.filterwarnings('ignore')





class BatchSampler(Sampler):
    """
    Batch-specific Sampler
    sampled data of each batch is from the same dataset.
    """
    def __init__(self, batch_size, batch_id, drop_last=False):
        """
        create a BatchSampler object
        
        Parameters
        ----------
        batch_size
            batch size for each sampling
        batch_id
            batch id of all samples
        drop_last
            drop the last samples that not up to one batch
            
        """
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_id = batch_id

    def __iter__(self):
        batch = {}
        sampler = np.random.permutation(len(self.batch_id))
        for idx in sampler:
            c = self.batch_id[idx]
            if c not in batch:
                batch[c] = []
            batch[c].append(idx)

            if len(batch[c]) == self.batch_size:
                yield batch[c]
                batch[c] = []

        for c in batch.keys():
            if len(batch[c]) > 0 and not self.drop_last:
                yield batch[c]
            
    def __len__(self):
        if self.drop_last:
            return len(self.batch_id) // self.batch_size
        else:
            return (len(self.batch_id)+self.batch_size-1) // self.batch_size
    
def create_batchind_dict(adata,batch_name='batch'):#max
    max_batchid=adata.obs[batch_name].value_counts().idxmax()
    batchind_list=adata.obs[batch_name].cat.categories.tolist()
    max_batchindex=batchind_list.index(max_batchid)
    batchind_list[max_batchindex]=batchind_list[0]
    batchind_list[0]=max_batchid
    for j in range(len(batchind_list)):
        batch_ind_dict={batchind_list[i]:i for i in range(len(batchind_list))}
    return batch_ind_dict

def create_batchind_list(adata,batch_name='batch'):#max
    max_batchid=adata.obs[batch_name].value_counts().idxmax()
    batchind_list=adata.obs[batch_name].cat.categories.tolist()
    max_batchindex=batchind_list.index(max_batchid)
    batchind_list[max_batchindex]=batchind_list[0]
    batchind_list[0]=max_batchid
   
    return batchind_list
        


    
# def create_dataloader(adata,batch_size,batchind_dict,batch_name='batch',num_worker=4,droplast=False):
#     scdata = SingleCellDataset(adata,batchind_dict,batch_name) 
#     scdataloader = DataLoader(scdata, batch_size=batch_size,num_workers=num_worker,drop_last=droplast,shuffle=True)
#     return scdataloader
    
class SingleCellDataset(Dataset):
    """
    Dataloader of single-cell data
    """
    def __init__(self, adata, batchind_dict, batch_name='batch'):
        """
        Create a SingleCellDataset object
        
        Parameters
        ----------
        adata
            AnnData object wrapping the single-cell data matrix
        """
        self.adata = adata
        self.adata_batch = adata.obs[batch_name]
        self.batchind_dict = batchind_dict
        self.batch_name = batch_name
        
        

    def __len__(self):
        return self.adata.shape[0]
    
    def __getitem__(self, idx):
        
        x = torch.tensor(self.adata.X[idx].toarray().squeeze())

        # 获取 domain_id
        domain_id = self.adata_batch[idx]
        domain_id = self.batchind_dict[domain_id]
        
        return x, domain_id, idx

    

  
    
    
    
def create_dataloader(adata, batch_size, batchind_dict, batch_name='batch', num_worker=4, droplast=False):
    scdata = SingleCellDataset(adata, batchind_dict, batch_name) 
    scdataloader = DataLoader(scdata, batch_size=batch_size, num_workers=num_worker, drop_last=droplast, shuffle=True)
    return scdataloader





