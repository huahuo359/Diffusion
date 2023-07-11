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

from model.unet import UNet 
from model.ddpm import DDPM   
from dataset.kaggle import MyDataSet 
from dataset.kaggle import SubsetSampler

import matplotlib.pyplot as plt
import math

import argparse
import random

class Configs: 
    
    def __init__(
        self, 
        image_size = 64, 
        image_channel = 3,
        n_channels = 64, # 第一次卷积的输出通道数
        ch_mults = (1, 2, 2, 4), 
        is_atten = (False, False,True, True), 
        n_steps = 1000, 
        batch_size = 16,
        n_samples = 4,
        learning_rate = 2e-5 ,
        epochs = 10, 
        n_blocks = 2, 
        root = './data/', 
        img_dir = 'kaggle/', 
        data_txt = 'name.txt',
        load = True, 
        net_name = 'Gen8000.pth'
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
        
        # current_file_path = os.path.abspath(__file__)
        # current_directory = os.path.dirname(current_file_path)
        # new_directory = current_directory + '/data/'
        # print("dir:", new_directory)
        # root = new_directory
       
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

        self.dataset = MyDataSet(root=root, img_dir=img_dir, datatxt=data_txt, transform=transform)

        # 指定每个 epoch 中获取的数据量
        samples_per_epoch = 8192

        # 创建自定义采样器
        indices = torch.randperm(len(self.dataset))[:samples_per_epoch]
        sampler = SubsetSampler(indices)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, sampler=sampler)

        
        self.NetWork = UNet(
            image_channels = self.image_channel,
            n_channels = self.n_channels,
            n_blocks = self.n_blocks,
            ch_mults = self.ch_mults,
            is_atten = self.is_atten
        )
        
        self.net_name = net_name
        
        if load is True: 
            print('[INFO] load the pre-trained network: ', net_name)
            self.NetWork.load_state_dict(torch.load(net_name))
        else:
            print('[INFO] train a new network: ', net_name)
            
            
        
        # self.NetWork.load_state_dict(torch.load('Gen8000.pth'))
        
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
            
            if i % 32 == 31: 
                print(f'[INFO] running_loss [{i + 1:5d}] loss: {running_loss / 32:.3f}')
                running_loss = 0.0
                
                
    def run(self): 
        torch.save(self.NetWork.state_dict(), self.net_name)
        
        for epoch in range(self.epochs): 
            print('[INFO] epoch: ', epoch)
            
            self.train_one_epoch() 
            self.show_generate(title = "Gen"+str(epoch))
           
            torch.save(self.NetWork.state_dict(), self.net_name)
            
        
        
    def show_images(self,images, title="show images"): 
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()
        
        fig = plt.figure(figsize=(10, 10))
        img_num = len(images)
        rows = int(math.sqrt(img_num))
        cols = round(img_num / rows)
        
        rows = 2 
        cols = 2

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
            
                    
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser() 
    parser.add_argument('--b', type=int, default=32, help='input batch size')
    parser.add_argument('--image_size', type=int, default=32, help='the size of generate images') 
    parser.add_argument('--channel', type=int, default=3, help='the channel of images')
    parser.add_argument('--ch_mults', nargs=3, help='args for unet: ch_mults, for single unet the number is 3' )
    parser.add_argument('--n', type=int, default=1, help='number of down/up block for unet')
    parser.add_argument('--step', type=int, default=1000, help='the steps of adding noise/denoise') 
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--epoch', type=int, default=32, help='epoches to train')
    parser.add_argument('--dataset', type=str, required=True, help='dataset path')
    parser.add_argument('--img_dir', type=str, required=True, help='choose the dir for trained dir')
    parser.add_argument('--data_txt', type=str, help='choose data.txt for dataset')
    parser.add_argument('--load', action="store_true", help='load the pre-trained model or not')
    parser.add_argument('--model_name', type=str, help='the name for trained/pre-trained network')
    
    
    args = parser.parse_args() 
    args.seed = random.randint(1, 100)  # set random seed for add noise
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


    return args      
            
            
def main(): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('[INFO] device selected: ', device)
    
    args = parse_args() 
    print('[INFO] set configs for training')
    
    configs = Configs(image_size = args.image_size,
                      image_channel = args.channel, 
                      epochs = args.epoch,
                      batch_size = args.b, 
                      ch_mults = (1, 2, 3, 4), 
                      is_atten = (False,False, True, True), 
                      root = args.dataset,
                      img_dir = args.img_dir, 
                      data_txt = args.data_txt,
                      load=args.load, 
                      net_name=args.model_name
                      )
    # configs.show_x0() 
    print('[INFO] start to train')
    configs.run() 
    # configs.show_generate(title="G2")

if __name__ == '__main__':
    main()