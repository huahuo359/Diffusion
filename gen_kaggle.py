import sys
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import datasets

from model.unet import UNet 
from model.ddpm import *   
from model.ddim import *
from dataset.kaggle import MyDataSet 
from dataset.kaggle import SubsetSampler

import matplotlib.pyplot as plt
import math
import imageio

import argparse
import random

import time
import numpy as np

class Configs: 
    
    def __init__(
        self, 
        image_size = 64, 
        image_channel = 3,
        n_channels = 64, # The number of output channels of the first convolution
        ch_mults = (1, 2, 2, 4), 
        is_atten = (False, False,True, True), 
        n_steps = 1000, 
        batch_size = 1,
        n_samples = 4,
        learning_rate = 2e-5 ,
        epochs = 10, 
        n_blocks = 2, 
        ddim_steps = 100,
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
        self.ddim_steps = ddim_steps
       
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
       
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

        
        self.ddpm = DDPM(
            network = self.NetWork,
            steps = self.n_steps,
            device = self.device, 
        )
        
        self.ddim = DDIM(
            network = self.NetWork,
            steps = self.n_steps,
            device = self.device, 
        )
        
            
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
                    plt.imshow( images[index] ,vmin=0, vmax=1)
                    index += 1
        # fig.suptitle('name',fontsize=30)      
        plt.savefig(f"{title}.png")
        
        plt.show()
        
    def show_x0(self): 
        for batch in self.dataloader: 
            
            batch = batch.permute(0, 2, 3, 1) 
            self.show_images(batch, "x0")
            
            break
    
    def show_generate_ddpm(self, title="Gen_ddpm"): 
        
        with torch.no_grad(): 
            x = torch.randn([self.n_samples, self.image_channel, self.image_size, self.image_size],device=self.device)
            
            x = self.ddpm.gen_img(x)    
            x = x.permute(0, 2, 3, 1) 
            self.show_images(x, title)
            
    def show_generate_ddim(self, title="Gen_ddim"): 
        
        with torch.no_grad(): 
            x = torch.randn([self.n_samples, self.image_channel, self.image_size, self.image_size],device=self.device)
            
            x = self.ddim.gen_img(xt=x, ddim_step=self.ddim_steps)    
            x = x.permute(0, 2, 3, 1) 
            self.show_images(x, title)
            
    def show_step(self, title="Step"): 
        # show the steps of generating an image
        with torch.no_grad(): 
            x = torch.randn([1, self.image_channel, self.image_size, self.image_size],device=self.device)
            
            
            gen_steps = [] # Save generated intermediate steps
            pil_images = []
            
            for t_ in range(self.n_steps): 
                t = self.n_steps - t_ - 1 
                x = self.ddpm.p_sample(x, x.new_full((1,), t, dtype=torch.long))
                if t_%100 == 0 or (t_>700 and t_%40 ==0 ) or (t_>900 and t_%20 ==0 ) or t_==990: # steps is 1000 
                   
                    gen_steps.append(x.permute(0, 2, 3, 1).clone().detach().cpu()) 
                    step_img = x.permute(0, 2, 3, 1).clone().detach().cpu().numpy()
                    step_img = np.uint8(step_img*255)  # Convert to uint8
                    step_img = step_img.squeeze()  # Flatten the array
                    # step_img.resize((300,300))
                    pil_images.append(Image.fromarray(step_img))
                
            
         
            gen_steps.append(x.permute(0, 2, 3, 1).clone().detach().cpu().numpy())
            
            step_img = x.permute(0, 2, 3, 1).clone().detach().cpu().numpy()
            step_img = np.uint8(step_img*255)  # Convert to uint8
            step_img = step_img.squeeze()  # Flatten the array
           
            pil_images.append(Image.fromarray(step_img))
        
        # Create the GIF using imageio
        # pil_images = [Image.fromarray(img[0]) for img in gen_steps]

        imageio.mimsave("generated_steps.gif", pil_images, duration=200)
        fig = plt.figure(figsize=(32, 8))
        img_num = len(gen_steps)
    
        rows = 2
        cols = img_num // 2

        index = 0
        
        for i in range(rows):
            for j in range(cols):
                fig.add_subplot(rows,cols,index+1)
                
                if index < img_num:
                    plt.imshow(gen_steps[index][0])
                    index += 1
                    
        plt.savefig(f"{title}.png")
        
        plt.show()
    
    
    def create_gif(images, filename, duration=200):
        # Create GIF from a list of images
        images[0].save(filename, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)
    
    
    def show_inverse(self, title="Inverse"):
        with torch.no_grad:
            for i, data in enumerate(self.dataloader, 0): 
                inputs = data 
                inputs = inputs.to(self.device)
                self.show_images(title="start_x0")
                noise_in = self.ddpm.q_sample(inputs, 999)
                self.show_images(title="noise_x0")
                
            x = noise_in
            
            
            gen_steps = [] # Save generated intermediate steps
            
            for t_ in range(self.n_steps): 
                t = self.n_steps - t_ - 1 
                x = self.ddpm.p_sample(x, x.new_full((1,), t, dtype=torch.long))
                if t_%100 == 0 or (t_>700 and t_%40 ==0 ) or (t_>900 and t_%20 ==0 ) or t_==990: # steps is 1000 
                    gen_steps.append(x.permute(0, 2, 3, 1).clone().detach().cpu()) 
                
                    
            gen_steps.append(x.permute(0, 2, 3, 1).clone().detach().cpu().numpy())
        
        fig = plt.figure(figsize=(32, 8))
        img_num = len(gen_steps)
    
        rows = 2
        cols = img_num // 2

        index = 0
        
        for i in range(rows):
            for j in range(cols):
                fig.add_subplot(rows,cols,index+1)
                
                if index < img_num:
                    plt.imshow(gen_steps[index][0])
                    index += 1
                    
        plt.savefig(f"{title}.png")
        
        plt.show()
                
                    
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
    parser.add_argument('--gen_name', type=str, help='the generate image name')
    parser.add_argument('--ddim_steps', type=int, help='the steps for ddim to generate picture')
    
    
    
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
                      ddim_steps = args.ddim_steps,
                      ch_mults = (1, 2, 3, 4), 
                      is_atten = (False,False, True, True), 
                      root = args.dataset,
                      img_dir = args.img_dir, 
                      data_txt = args.data_txt,
                      load=args.load, 
                      net_name=args.model_name
                      )
    configs.show_x0() 
    print('[INFO] start to gen image by ddpm')
    start_time = time.time()

    configs.show_generate_ddpm(title=(args.gen_name+'_ddpm'))

    end_time = time.time()
    elapsed_time_ddpm = end_time - start_time

    start_time = time.time()
    print("[INFO] Elapsed time for show_generate_ddpm:", elapsed_time_ddpm, "seconds")
    
    print('[INFO] start to gen image by ddim')
    configs.show_generate_ddim(title=(args.gen_name+'_ddim'))

    end_time = time.time()
    elapsed_time_ddim = end_time - start_time
    print("[INFO] Elapsed time for show_generate_ddim:", elapsed_time_ddim, "seconds")

    print("[INFO] start to show steps to generate an image by ddpm")
    configs.show_step()


if __name__ == '__main__':
    main()