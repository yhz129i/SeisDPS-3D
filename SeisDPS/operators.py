import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import convolution_matrix

class SeisOp(nn.Module):
    def __init__(self, wavelet , Nref, sv_threshold=0.04792004802863965, dtype=torch.cuda.FloatTensor):
        """Pytorch implementation of forward/pseudo-inverse operators for the deconvolution problem. 

        Args:
            wavelet (1D numpy array): seismic wavelet
            Nref (integer): lenght of reflectivity model
            sv_threshold (float, optional): kill singular values of normalized sigma below sv_threshold. Defaults to 0..
            dtype: Defaults to torch.cuda.FloatTensor.
        """
        super(SeisOp, self).__init__()
        self.dtype = dtype
        
        H = convolution_matrix(wavelet, Nref, mode='same')
        # HT = np.linalg.pinv(H, rcond=sv_threshold)
        HT = np.linalg.pinv(H)
        
        self.H = torch.from_numpy(H).type(self.dtype)
        self.HT = torch.from_numpy(HT).type(self.dtype)
    
    def forward(self, x):
        return self.Convolve(x)
    
    def Convolve(self, x):
        assert len(x.shape)==4, '4D input: NCHW'
        y = torch.matmul(self.H, x)           
        return y.type(self.dtype)
    
    def Deconvolve(self, y):
        x = torch.matmul(self.HT,y)
        return x.type(self.dtype)
    
    def Forward(self, x):
        return self.Convolve(x)
    
    def Backproj(self, y):
        return self.Deconvolve(y)
    
    def Pr(self, x):
        return self.Backproj(self.Forward(x))
    
    def Pn(self, x):
        return x - self.Pr(x)
    
    def Convolve_H_T(self, x):
        assert len(x.shape)==4, '4D input: NCHW'
        y = torch.matmul(self.H.T, x)           
        return y.type(self.dtype)
    
    def Convolve_H_H(self, x):
        assert len(x.shape)==4, '4D input: NCHW'
        H_H = torch.conj(self.H.T)
        y = torch.matmul(H_H, x)           
        return y.type(self.dtype)
    
