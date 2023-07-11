import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import math

class TimeEmbedding(nn.Module): 
    
    def __init__(self, n_channels): 
        super().__init__()
        self.n_channels = n_channels
        self.fc1 = nn.Linear(n_channels // 4, n_channels)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(n_channels, n_channels) 
        
    def forward(self, t): 
        half_dim = self.n_channels // 8 
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        batches = emb.shape[0]
        emb = emb.reshape(batches, -1)

        emb = self.act(self.fc1(emb))
        emb = self.fc2(emb)
        
        return emb
    
    
class ResBlock(nn.Module): 
    
    def __init__(self, in_channels, out_channels, time_channels, n_groups=16, drop_out = 0.1):
        super().__init__()  
        self.norm1 = nn.GroupNorm(n_groups, in_channels) 
        self.act1 = nn.SiLU() 
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) 
        
        self.norm2 = nn.GroupNorm(n_groups, out_channels) 
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  
        
        self.cross = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels!=out_channels else nn.Identity() 
        
        self.time_emb = nn.Linear(time_channels, out_channels) 
        self.time_act = nn.SiLU()
        
        self.dropout = nn.Dropout(drop_out)
        
    def forward(self, x, t): 
        
        out = self.conv1(self.act1(self.norm1(x)))
        out += self.time_emb(self.time_act(t))[:, :, None, None]
        out = self.conv2(self.dropout(self.act2(self.norm2(out))))

        return out + self.cross(x)
        
        
class SelfAttention(nn.Module): 
    
    def __init__(self, in_dim, heads=8, dim_head = None, n_groups = 32): 
        # in_dim: 输入的通道数, 和 forward 中的 c 相同
        super().__init__() 
        
        if dim_head == None: 
            dim_head = in_dim  
        
        self.dim_head = dim_head
        self.heads = heads 
        self.scale = dim_head ** -0.5
        
        self.norm = nn.GroupNorm(n_groups, in_dim) 
        self.to_qkv = nn.Linear(in_dim, 3*heads*dim_head) 
        self.output = nn.Linear(heads*dim_head, in_dim)  
        
        
    def forward(self, x, t=None): 
        
        b, c, h, w = x.shape # bach_size * channel * height * width
        
        x = x.view(b, c, -1).permute(0, 2, 1)   # change shape to b*(h*w)*c  
        
        qkv = self.to_qkv(x).view(b, -1, self.heads, 3*self.dim_head)
        
        q,k,v = torch.chunk(qkv, 3, dim=-1)  
        
        dots = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale 
        attn = dots.softmax(dim=2) 
        
        res = torch.einsum('bijh,bjhd->bihd', attn, v) 
        res = res.reshape(b, -1, self.heads * self.dim_head) 
        res = self.output(res) 
        
        res += x  # shape: b*(n)*c  
        
        res = res.permute(0, 2, 1).view(b, c, h, w) 
        
        return res  
    
    
        
class DownBlock(nn.Module): 
    
    def __init__(self, in_channels, out_channels, time_channels, has_atten):
        
        super().__init__() 
        self.resblock = ResBlock(in_channels, out_channels, time_channels) 
        self.attnblock = SelfAttention(out_channels) if has_atten else nn.Identity() 
            
        
    def forward(self, x, t): 
        x = self.resblock(x, t) 
        x = self.attnblock(x) 
        return x  
    
class UpBlock(nn.Module): 
    
    def __init__(self, in_channels, out_channels, time_channels, has_atten): 
        
        super().__init__() 
        # UpBlock 实现跳跃连接
        self.resblock = ResBlock(in_channels+out_channels, out_channels, time_channels) 
        self.attnblock = SelfAttention(out_channels) if has_atten else nn.Identity() 
        
    
    def forward(self, x, t): 
        x = self.resblock(x, t) 
        x = self.attnblock(x) 
        return x 
        
        
        
class MiddleBlock(nn.Module): 
    
    def __init__(self, in_channels, time_channels): 
        
        super().__init__() 
        self.res1 = ResBlock(in_channels, in_channels, time_channels) 
        self.atten = SelfAttention(in_channels)
        self.res2 = ResBlock(in_channels, in_channels, time_channels) 
        
    def forward(self, x, t): 
        x = self.res1(x, t) 
        x = self.atten(x) 
        x = self.res2(x, t)  
        return x  
    
class Upsample(nn.Module): 
    # 把图像的尺寸扩大二倍
    def __init__(self, in_channels): 
        super().__init__() 
        self.upblock = nn.ConvTranspose2d(in_channels, in_channels,4,2,1)
        
    def forward(self, x, t): 
        
        return self.upblock(x)   
    
class Downsample(nn.Module): 
    # 把图像的尺寸缩减一半
    def __init__(self, in_channels): 
        super().__init__() 
        self.downblock = nn.Conv2d(in_channels, in_channels, 3,2,1) 
        
    def forward(self, x, t): 
        
        return self.downblock(x)  
    
    
class UNet(nn.Module): 
    
    def __init__(
        self, 
        image_channels = 3, 
        n_channels = 64, 
        n_blocks = 2,   # DownBlock/UpBlock 重复次数
        ch_mults = (1, 2, 2, 4), 
        is_atten = (False, False, True, True)
    ): 
        super().__init__() 
        self.conv1 = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)
        self.embblock = TimeEmbedding(4*n_channels) 
        
        down_blocks = [] 
        
        in_channels = n_channels 
        out_channels = n_channels 
        
        n_res = len(ch_mults) 
        
        for i in range(n_res): 
            out_channels = in_channels * ch_mults[i] 
            
            for _ in range(n_blocks): 
                down_blocks.append(DownBlock(in_channels, out_channels, 4*n_channels, is_atten[i]))
                in_channels = out_channels 
                
            
            if i < n_res-1: 
                down_blocks.append(Downsample(out_channels))
                
        
        
        self.down_blocks = nn.ModuleList(down_blocks) 
        
        self.midblock = MiddleBlock(out_channels, 4*n_channels)  
        
        
        up_blocks = [] 
        
        in_channels = out_channels 
        
        for i in reversed(range(n_res)): 
            
            out_channels = in_channels
            
            for _ in range(n_blocks): 
                up_blocks.append(UpBlock(in_channels, out_channels, 4*n_channels, is_atten[i]))
                
            out_channels = in_channels // ch_mults[i]
            up_blocks.append(UpBlock(in_channels, out_channels, 4*n_channels, is_atten[i]))
            in_channels = out_channels
    
            if i>0: 
                up_blocks.append(Upsample(out_channels))
                
        self.up_blocks = nn.ModuleList(up_blocks)
        
        self.to_out = nn.Sequential(
            nn.GroupNorm(8, n_channels),  
            nn.SiLU(), 
            nn.Conv2d(out_channels, image_channels, kernel_size=3, padding=1)
        )
        
       
        
    
    def forward(self,x, t): 
        
        t = self.embblock(t) 
        x = self.conv1(x) 
        
        x_steps = [x]
        
        for downblock in self.down_blocks: 
            x = downblock(x, t)
            x_steps.append(x)       
            
        x = self.midblock(x, t)  
        
        for upblock in self.up_blocks: 
            
            if isinstance(upblock, Upsample): 
                x = upblock(x,t) 
            else: 
                revers_x = x_steps.pop() 
                x = torch.cat((x, revers_x), dim=1)      
                x = upblock(x, t)       
                
        return self.to_out(x)