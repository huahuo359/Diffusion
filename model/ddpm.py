import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import math

def linear_beta_schedule(steps=1000, min_beta=0.001, max_beta=0.02):
    return torch.linspace(min_beta, max_beta, steps)
    
def cosine_beta_schedule(steps, s = 0.008): 
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    x = torch.linspace(0, steps, steps+1)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPM(nn.Module):
    def __init__(self, 
                 network, 
                 steps=1000, 
                 beta_schedule = 'linear',
                 min_beta=0.0001, 
                 max_beta=0.02, 
                 device=None, 
                 img_size=(3, 64, 64)):
        super().__init__()
        self.network = network.to(device)   # 训练好的去除噪声的神经网络
        self.steps = steps
        self.device = device
        self.img_size = img_size
        
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(steps, min_beta, max_beta)
            self.betas = betas.to(device) 
        elif beta_schedule == 'cosine': 
            betas = cosine_beta_schedule(steps) 
            self.betas = betas.to(device)
            
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)  # 累计项的乘积
        self.sigma2 = self.betas 
    
    
    def q_sample(self, x0, t, eta=None): 
        # add noise 
        b, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        if eta is None:  
            eta = torch.randn(b, c, h, w).to(self.device) 
        
        noise = a_bar.sqrt().reshape(b, 1, 1, 1)*x0 + (1-a_bar).sqrt().reshape(b, 1, 1, 1)*eta
        return noise
        
        
    def p_sample(self, xt, t): 
        # 去除噪声 
        
        eta_theta = self.network(xt, t) # 预测出的可能添加的噪声
    
        a_bar = self.alpha_bars[t]
        alpha = self.alphas[t] 
        
        eta_coef = (1-alpha)/((1-a_bar)**0.5)
        batch_size = eta_coef.shape[0]
       
       
        test = eta_coef.view(batch_size, 1, 1, 1) * eta_theta
      
        mean = 1/(alpha.view(batch_size, 1, 1, 1)**0.5)*(xt - test)
        
        sigma2 = self.sigma2[t] 
        
        b, c, h, w = xt.shape
        eta = torch.randn(b, c, h, w).to(self.device)
        
        predict = mean + (sigma2.view(batch_size, 1, 1, 1)**0.5)*eta  
        
        return predict 
    
    def predict_x0(self, xt, t): 
        a_bar = self.alpha_bars[t]
        predict_noise = self.network(xt, t) 
        
        predict = (xt - (1-a_bar)**0.5 * predict_noise) / (a_bar ** 0.5)
    
    
  
    def loss(self, x0, noise=None): 
        
        # Computes the loss of the noise generated versus the noise actually added
        batch_size = x0.shape[0] 
        t = torch.randint(0, self.steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # generate the noise
        if noise is None: 
            noise = torch.randn_like(x0) 
        
        # enerate noisy Image
        xt = self.q_sample(x0, t, noise) 
        
        pre_noise = self.network(xt, t)   
        
        return F.mse_loss(noise, pre_noise, reduction='sum')  
    
    def gen_img(self, xt):
        n_samples = xt.shape[0]
        for t_ in range(self.steps):
            t = self.steps - t_ - 1 
            xt = self.p_sample(xt, xt.new_full((n_samples,), t, dtype=torch.long))          
        
        return xt
    
