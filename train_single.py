import sys
import os
# current_file_path = os.path.abspath(__file__)
# current_directory = os.path.dirname(current_file_path)
# root = current_directory + '/'
# print(root)

# sys.path.append(root)

import torch
import torch.utils.data
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets


from model.sunet import UNet 
from model.ddpm import DDPM   
from dataset.single import MyDataSet 
from dataset.single import SubsetSampler

import torch, gc

import matplotlib.pyplot as plt
import math


class Configs: 
    
    def __init__(
        self, 
        image_size = 128, 
        image_channel = 3,
        n_channels = 64, # 第一次卷积的输出通道数
        ch_mults = (1, 2), 
        is_atten = (False, False), 
        n_steps = 1000, 
        batch_size = 32,
        n_samples = 16,
        learning_rate = 2e-5 ,
        epochs = 10, 
        n_blocks = 1
        ): 
        
        self.image_size = image_size
        self.image_channel = image_channel
        self.n_channels = n_channels 
        self.ch_mults = ch_mults 
        self.is_atten = is_atten 
        self.n_steps = n_steps 
        self.batch_size = batch_size  
        self.n_samples = n_samples 
        self.lr = learning_rate 
        self.epochs = epochs 
        self.n_blocks = n_blocks
       
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        new_directory = current_directory + '/data/'
        print("dir:", new_directory)
        root = new_directory

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        self.dataset = MyDataSet(root=root, filename='geese.png', transform=transform)

        # 指定每个 epoch 中获取的数据量
        samples_per_epoch = 8000

        # 创建自定义采样器
        indices = torch.randperm(len(self.dataset))[:samples_per_epoch]
        sampler = SubsetSampler(indices)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, sampler=sampler)

        
        self.NetWork = UNet(
            image_size= 128,
            image_channels = self.image_channel,
            n_channels = self.n_channels,
            n_blocks = self.n_blocks,
            ch_mults = self.ch_mults,
            is_atten = self.is_atten
        )
        
        # self.NetWork.load_state_dict(torch.load('geese.pth'))
        
        self.ddpm = DDPM(
            network = self.NetWork,
            steps = self.n_steps,
            device = self.device, 
        )
        
        self.opt = torch.optim.Adam(self.NetWork.parameters(), lr=self.lr)
        
        
    def train_one_epoch(self): 
        
        running_loss = 0.0 
        
        for i, data in enumerate(self.dataloader, 0): 
            inputs = data 
            inputs = inputs.to(self.device)
            
            self.opt.zero_grad() 
            loss = self.ddpm.loss(inputs)  
            loss.backward() 
            self.opt.step()   
            
            running_loss += loss.item()
            if i % 16 == 15: 
                print(f'[{i + 1:5d}] loss: {running_loss / 16:.3f}')
                running_loss = 0.0
                
                
    def run(self): 
        
        for epoch in range(self.epochs): 
            print('epoch: ', epoch)
            self.train_one_epoch() 
            self.show_generate(title = "Gen"+str(epoch))
            torch.save(self.NetWork.state_dict(), 'geese.pth')
        
    def show_images(self,images, title="show images"): 
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()
        
        fig = plt.figure(figsize=(16, 16))
        img_num = len(images)
        rows = int(math.sqrt(img_num))
        cols = round(img_num / rows)
        
        rows = 4
        cols = 4

        index = 0
        
        for i in range(rows):
            for j in range(cols):
                fig.add_subplot(rows,cols,index+1)
                
                if index < img_num:
                    plt.imshow(images[index])
                    index += 1
                    
        fig.suptitle(title,fontsize=30)
        plt.savefig(f"{title}.png")
        
        plt.show()
        
    def show_x0(self): 
        for batch in self.dataloader: 
            
            batch = batch.permute(0, 2, 3, 1) 
            self.show_images(batch, "x0")
            
            print(batch[0].shape)
            break
    
    def show_generate(self, title="Gen"): 
        
        with torch.no_grad(): 
            x = torch.randn([self.n_samples, self.image_channel, self.image_size, self.image_size],device=self.device)
        
            for t_ in range(self.n_steps):
                t = self.n_steps - t_ - 1 
                x = self.ddpm.p_sample(x, x.new_full((self.n_samples,), t, dtype=torch.long))        
            x = x.permute(0, 2, 3, 1) 
            self.show_images(x, title)
            
def main(): 
    
    
    configs = Configs(image_size=128,image_channel=3, ch_mults = (1, 2, 4), 
        is_atten = (False, False, False))
    configs.run() 

if __name__ == '__main__':
    device = 'cuda'
    gc.collect()
    torch.cuda.empty_cache()
    main()



