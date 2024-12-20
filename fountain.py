import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
import pandas as pd 
from scipy.sparse import csr_matrix
import numpy as np
import torch.autograd as autograd
import itertools
from tqdm import tqdm
from collections import defaultdict
from itertools import cycle
from torch.autograd import Variable
import random
import math
import sys
import time
import anndata as ad
from Fountain.layer import *
from Fountain.loss import *
from Fountain.utils import *
from Fountain.data import *


class Fountain(nn.Module):

    
    def __init__(self,adata, enc, dec, n_domain,r_negative_multinomial=1.0,batch_name='batch'):
        """
        Parameters
        ----------
        enc
            Encoder structure config
        dec
            Decoder structure config
        
        n_domain
            The number of batches
        """
        super().__init__()
        
        self.z_dim = enc[-1][1]
        
        n_batch=n_domain
        x_dim=dec[-1][1]
        self.n_cell=adata.shape[0]
        self.encoder = Encoder_vae(x_dim, enc)
        self.decoder = NN(self.z_dim+n_batch, dec)
        self.n_domain = n_domain       
        self.batch_ind=create_batchind_list(adata,batch_name)        
        self.r=nn.Parameter(torch.Tensor([r_negative_multinomial]),requires_grad=True) 
        del adata    
        
    def get_latent(self,dataloader,device='cuda:0'):
        self.to(device)
        n_cell=len(dataloader.dataset)
        X_latent=torch.zeros((n_cell, self.z_dim)).to(device)
        for x,domain_id,idx in dataloader:
            with torch.no_grad():
                X=x.float().to(device) 
                mu=self.encoder.forward(X)[1]
                X_latent[idx]=mu
        return  X_latent.cpu().numpy()
            
        
    
    def enhance(self,adata,device='cuda:0',latent_rep='mu',batch_name='batch'):
        X=torch.tensor(adata.X.todense()) .to(device) 
        self.to(device)
        z,mu,_=self.encoder(X)
        if latent_rep=="mu":
            rep=mu            
        elif latent_rep=="z":           
            rep=z
        batch_token=(0*torch.ones(z.size(0))).long().to(device)
        batch_token=F.one_hot(batch_token,num_classes=self.n_domain).to(torch.float32)
        recon_input=torch.cat((rep,batch_token),1)
        recon = self.decoder(recon_input)
        zero=torch.zeros((recon.size(0), 1),device=device)
        recon_cat=torch.cat((zero,recon),dim=1)
        P=torch.nn.Softmax(dim=1)(recon_cat+1e-10)
        p_0=P[:,0:1]    
        p=P[:,1:]
        r=((self.r)**2)*torch.ones_like(p_0)
        count=(r/p_0)*p       
        Enhance=torch.where(count > 1.0, torch.tensor(1), torch.tensor(0))
        enh=Enhance+X
        Enhance_out=torch.where( enh > 0.9, torch.tensor(1), torch.tensor(0))
        return csr_matrix(Enhance_out.detach().cpu().numpy()  , dtype=np.float32)

    
    def train(
            self,
            dataloader,             
            lambda_mse=0.005, 
            lambda_Eigenvalue=0.5,
            lambda_kl=0.5,
            lambda_recon=1.0, 
            reg=0.1,
            reg_m=5.0,
            lr_pre=1e-4,
            lr=2e-4,
            lr_r=5e-4,            
            max_iteration=30000,
            mid_iteration=3000,
            ot_iteration={'outer':50,'inner':2},
            knn_neighbors=10,
            eigenvalue_type='mean',
            loss='Negative_multinomial',            
            t=5.0,
            early_stopping=None,                        
            device='cuda:0',
            batch_name='batch',
            verbose=False,
        ):
       

        self.to(device)        
        lambda_L2=torch.tensor(0.0).to(device)  
        n_epoch = int(np.ceil(max_iteration/len(dataloader)))
        mid_epoch=int(np.ceil(mid_iteration/len(dataloader)))
        damp_iter=int(np.ceil(   (n_epoch-mid_epoch)/15    )   )
        if loss=='Negative_multinomial':
            loss_func=Negative_multinomial_loss
            optim = torch.optim.Adam([{'params': itertools.chain(self.encoder.parameters(), self.decoder.parameters()),'lr': lr_pre,'weight_decay':5e-4 }, {'params': self.r, 'lr': lr_r,'weight_decay':5e-4}])            
        elif loss=='multinomial':
            loss_func=multinomial_loss
            optim = torch.optim.Adam(self.parameters(), lr=lr_pre, weight_decay=5e-4)
        
             
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:       
            for epoch in tq:
                
                if epoch>mid_epoch+1 and epoch<mid_epoch+damp_iter+2:
                    lambda_L2+=lambda_mse/damp_iter
                if epoch==mid_epoch+1:
                    
                    if loss=='Negative_multinomial' :
                        optim = torch.optim.Adam([{'params': itertools.chain(self.encoder.parameters(), self.decoder.parameters()),'lr': lr,'weight_decay':5e-4 }, {'params': self.r, 'lr': lr_r,'weight_decay':5e-4}])
                    
                    elif  loss=='multinomial':
                        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)

                tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations', disable=(not verbose))
                epoch_loss = defaultdict(float)
                                                                                 
                for i, (x,domain_id,_) in tk0:
                    
                    recon_loss = torch.tensor(0.0).to(device)
                    kl_loss = torch.tensor(0.0).to(device)
                    mse_loss= torch.tensor(0.0).to(device)

                    x= x.float().to(device)
                    domain_id=domain_id.long().to(device) 
                    if len(torch.unique(domain_id)) < self.n_domain:
                        print('Omit Some Batches')
                    batch_dict={}                  
                    z_dict={}
                    z_dict_new={}
                    for j in range(self.n_domain):
                        loc_query = torch.where(domain_id==j)[0]                            
                        X_j=x[loc_query]
                        if X_j.size(0)==0:
                            continue
                        z_j, mu_j, var_j = self.encoder(X_j)

                        z_dict[j] = z_j
                        if j>0 and epoch>mid_epoch:
                            if z_j.size(0)<=5:
                                print('Too few samples')
                                continue
                            ot_Matz=unbalanced_ot_z(z_dict[0],z_dict[j],reg=reg,reg_m=reg_m,device=device, max_iteration=ot_iteration)
                            if ot_Matz==None:
                                print('None OT')
                                
                                continue
                            L_j=Graph_Laplacian_torch(z_j,nearest_neighbor=min(knn_neighbors,z_dict[j].size(0)-1),t=t)
                            z_dict_new[j]=Transform(z_j,z_dict[0],torch.t(ot_Matz),L_j,lambda_Eigenvalue,eigenvalue_type)

                            mse_loss += F.mse_loss( z_dict_new[j], z_j ) * X_j.size(-1) 


                        batch_token=(j*torch.ones(z_j.size(0))).long().to(device)
                        batch_token=F.one_hot(batch_token,num_classes=self.n_domain).to(torch.float32)
                        recon_input=torch.cat((z_j,batch_token),1)
                        recon_j = self.decoder(recon_input)

                        if loss=='Negative_multinomial': 
                            rr=(self.r)**2                                
                            recon_loss += loss_func( recon_j, X_j,rr ) * X_j.size(-1) 
                        else:
                            recon_loss += loss_func( recon_j, X_j ) * X_j.size(-1)
                        kl_loss += kl_div(mu_j, var_j)

                    loss_value ={'recon_loss':lambda_recon*recon_loss,'kl_loss':lambda_kl*kl_loss,'mse_loss':lambda_L2*mse_loss} 
                    optim.zero_grad()
                    loss_sum=sum(loss_value.values())                      
                    loss_sum.requires_grad_(True)
                    loss_sum.backward()
                    optim.step()
                    
                    for k,v in loss_value.items():
                        epoch_loss[k] += loss_value[k].item()

                    info = ','.join(['{}={:.3f}'.format(k, v) for k,v in loss_value.items()])
                    tk0.set_postfix_str(info)

                epoch_loss = {k:v/(i+1) for k, v in epoch_loss.items()}
                epoch_info = ','.join(['{}={:.3f}'.format(k, v) for k,v in epoch_loss.items()])
                tq.set_postfix_str(epoch_info) 

                if early_stopping is not None:
                    early_stopping(sum(epoch_loss.values()), self)

                    if early_stopping.early_stop:
                        print('EarlyStopping: run {} epoch'.format(epoch+1))
                        break   
      
                                                    