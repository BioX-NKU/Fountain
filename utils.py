

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfTransformer
import scipy
import sklearn
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,MaxAbsScaler
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
def cluster_evaluation(adata_obs, label_key, cluster_key):
    '''
    Clustering Performance Evaluation

    Args:
        adata_obs: polars.internals.frame.DataFrame.
        label_key: e.g. 'cell type', 'cell_type'
        cluster_key: e.g. 'mc_Dleiden'

    Returns:
        evaluation

    '''
    print(cluster_key)
    AMI = sklearn.metrics.adjusted_mutual_info_score(adata_obs[label_key], adata_obs[cluster_key])
    ARI = sklearn.metrics.adjusted_rand_score(adata_obs[cluster_key], adata_obs[label_key])
    NMI = sklearn.metrics.normalized_mutual_info_score(adata_obs[cluster_key], adata_obs[label_key])
   
    return AMI,ARI,NMI









def overcorrection_score(emb, celltype, n_neighbors=100, n_pools=100, n_samples_per_pool=100, seed=124):
    n_neighbors = min(n_neighbors, len(emb) - 1)
    nne = NearestNeighbors(n_neighbors=1 + n_neighbors, n_jobs=8)
    nne.fit(emb)
    kmatrix = nne.kneighbors_graph(emb) - scipy.sparse.identity(emb.shape[0])

    score = 0
    celltype_ = np.unique(celltype)
    celltype_dict = celltype.value_counts().to_dict()
    
    N_celltype = len(celltype_)

    for t in range(n_pools):
        indices = np.random.choice(np.arange(emb.shape[0]), size=n_samples_per_pool, replace=False)
        score += np.mean([np.mean(celltype[kmatrix[i].nonzero()[1]][:min(celltype_dict[celltype[i]], n_neighbors)] == celltype[i]) for i in indices])

    return 1-score / float(n_pools)




def onehot(y, n):
    """
    Make the input tensor one hot tensors
    
    Parameters
    ----------
    y
        input tensors
    n
        number of classes
        
    Return
    ------
    Tensor
    """
    if (y is None) or (n<2):
        return None
    assert torch.max(y).item() < n
    y = y.view(y.size(0), 1)
    y_cat = torch.zeros(y.size(0), n).to(y.device)
    y_cat.scatter_(1, y.data, 1)
    return y_cat

def token_map(x,j,n_domain,device):
    #token is a one hot vector
    batch_token=(j*torch.ones(x.size(0))).to(torch.int64).to(device)
    batch_token=F.one_hot(batch_token,num_classes=n_domain).to(torch.float32)
    return torch.cat((x,batch_token),1)
        
        
        
class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, checkpoint_file=''):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 30
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss
        
class EarlyStopping_simple:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 30
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
       

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            
        elif score <= self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            self.best_score = score
           
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
      
        self.loss_min = loss   