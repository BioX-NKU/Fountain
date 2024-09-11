
import torch
import math
import numpy as np
from torch.distributions import Normal, kl_divergence
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph




def Negative_multinomial_loss(out,X,r):
    out+=1e-6
    device=out.device
    zero=torch.zeros((out.size(0), 1),device=device)
    if len(r)>1:
        vec_r=r.view(-1,1)
    else:
        vec_r=r*torch.ones((out.size(0), 1),device=device)
    out_cat=torch.cat((zero,out),dim=1)
    X_cat=torch.cat((vec_r,X),dim=1)
    log_p=torch.nn.LogSoftmax(dim=1)(out_cat+1e-8)
    loss=-torch.lgamma(torch.sum(X_cat,1)+1e-6) +torch.mean(torch.lgamma(vec_r+1e-6))- torch.sum(log_p*X_cat,1 )
    return loss.mean()/X.size(-1)

def multinomial_loss(out,X):
    out+=1e-6
    log_p=torch.nn.LogSoftmax(dim=1)(out)
    loss=-torch.sum(log_p*X,1 )
    return loss.mean()/X.size(-1)
    
    
    
def kl_div(mu, var):
    d=-0.5*(1.0+torch.log(var+1e-6)-mu**2-var)
    return torch.sum(d,dim=1).mean()

def kl_div_tst(mu, var):
    return kl_divergence(Normal(mu, var.sqrt()),
                         Normal(torch.zeros_like(mu),torch.ones_like(var))).sum(dim=1).mean()
 
def binary_cross_entropy(recon_x, x):
    return -torch.mean(x * torch.log(recon_x + 1e-8) + (1 - x) * torch.log(1 - recon_x + 1e-8), dim=-1) 



def clip_distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p, while restricting x_i,y_j in unit balls by multiplying a constant.

    Parameters
    ----------
    pts_src
        [R, D] matrix
    pts_dst
        [C, D] matrix
    p
        p-norm
    
    Return
    ------
    [R, C] matrix
         clipped distance matrix
    """
    clip_value=torch.tensor(1000.0)
    max_norm=torch.max(  torch.abs(pts_src).max()+  torch.abs(pts_dst).max() , 2*clip_value )/2
    pts_src_clip=clip_value*pts_src/max_norm
    pts_dst_clip=clip_value*pts_dst/max_norm
    x_col = pts_src_clip.unsqueeze(1)
    y_row = pts_dst_clip.unsqueeze(0)
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)
    return distance








def unbalanced_ot_z( x1, x2,reg=0.1, reg_m=1.0,device='cpu',max_iteration={'outer':10,'inner':5}):
    '''
    Calculate a unbalanced optimal transport matrix between mini subsets.
    Parameters
    ----------
    reg:
        Entropy regularization parameter in OT
    reg_m:
        Unbalanced OT parameter. Larger values means more balanced OT
    device
        training device

    Returns
    -------
    matrix
        mini-subset unbalanced optimal transport matrix
    '''
    outer_iter=max_iteration['outer']
    inner_iter=max_iteration['inner']
    ns = x1.size(0)
    nt = x2.size(0)
    cost_pp = clip_distance_matrix(x1, x2, p=2)
    
    p_s = torch.ones(ns, 1) / ns   
    p_t = torch.ones(nt, 1) / nt
    

    p_s = p_s.to(device)
    p_t = p_t.to(device)

    tran = torch.ones(ns, nt) / (ns * nt)
    tran = tran.to(device)

    dual = (torch.ones(ns, 1) / ns).to(device)
    f = reg_m / (reg_m + reg)

    for m in range(outer_iter):
        cost = cost_pp
        kernel = torch.exp(-cost / (reg*torch.max(torch.abs(cost)))) * tran
        b = p_t / (torch.t(kernel) @ dual)
        for i in range(inner_iter):
            dual =( p_s / (kernel @ b) )**f
            b = ( p_t / (torch.t(kernel) @ dual) )**f
        tran = (dual @ torch.t(b)) * kernel
        
    out=tran.detach()
    if torch.isnan(out).sum() > 0:        
        out=None
    return  out



def Graph_Laplacian_torch(X,nearest_neighbor=10,t=1.0):
    XX=X.detach()
    D=clip_distance_matrix(XX,XX)
    values,indices=torch.topk(D, nearest_neighbor+1, dim=1,largest=False)
    pos=D>values[:,nearest_neighbor].view(-1,1)
    D[pos] = 0.0
    graph_data=D
    graph_data = graph_data + graph_data.T.multiply(graph_data.T > graph_data) - graph_data.multiply(graph_data.T > graph_data)
    W =graph_data
    index_pos = torch.where(W>0)
    W_mean=torch.mean(W[index_pos])
    W[index_pos] =torch.exp(-W[index_pos]/(t*W_mean))
    return (torch.diag(W.sum(1)) - W).detach()



def Transform(X,Y,T,L,lamda_Eigenvalue,eigenvalue_type):
    Y=Y.detach()
    T=T.detach()
    L=L.detach()
    
    if eigenvalue_type=='mean':
        a=T.sum(1)
        a_inv=1.0/a               
        lamda_Lapalcian=2*lamda_Eigenvalue/(torch.diag(L)*a_inv).mean() #diag(ALA).mean()
        l=2*torch.mm(T,Y)        
        M=lamda_Lapalcian*L+2*torch.diag(a)
        if torch.isnan(M).sum() > 0:
            print('Laplacian NAN')
            M_inv= torch.diag(a_inv)/2.0
        else:
            M_inv=torch.linalg.inv(M).to(torch.float32)
        result=torch.mm(M_inv,l)
    
    elif eigenvalue_type=='normal':
        lamda_Lapalcian=lamda_Eigenvalue
        l=2*torch.mm(T,Y)        
        M=lamda_Lapalcian*L+2*torch.diag(T.sum(1))
        M_inv=torch.linalg.inv(M).to(torch.float32)
        result=torch.mm(M_inv,l)
    return result


