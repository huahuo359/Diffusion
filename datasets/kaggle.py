from PIL import Image 
import torch
import torchvision.transforms as transforms 
from torch.utils.data import Dataset, DataLoader, Sampler
from matplotlib import pyplot as plt 
import math 
import os 



def show_images(images, title="show images"): 
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()

        fig = plt.figure(figsize=(4, 4))
        img_num = len(images)
        rows = int(math.sqrt(img_num))
        cols = round(img_num / rows)

        index = 0
        
        for i in range(rows):
            for j in range(cols):
                fig.add_subplot(rows,cols,index+1)
                
                if index < img_num:
                    plt.imshow(images[index])
                    index += 1
                    
        fig.suptitle(title,fontsize=30)
        
        plt.show()

class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class MyDataSet(torch.utils.data.Dataset): 
    def __init__(self,root, datatxt, transform=None): 
        
        self.root = root
        fh = open(root + datatxt, 'r') 
        imgs = [] 
        
        for line in fh: 
            line = line.rstrip() 
            words = line.split(',') 
            imgs.append(words[0])
    
            
        self.imgs = imgs 
        self.transform = transform 
        
    def __getitem__(self, index): 
        fn = self.imgs[index]  
        fn = 'kaggle/'+fn
        img = Image.open(self.root+fn).convert('RGB')
        
        if self.transform is not None: 
            img = self.transform(img) 
        
        return img  
    
    def __len__(self): 
        return len(self.imgs)
    
    
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
new_directory = current_directory[:-len("dataset")]
new_directory = new_directory + 'data/'
print("dir:", new_directory)


root = new_directory
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data = MyDataSet(root=root, datatxt='name.txt', transform=transform)

# 指定每个 epoch 中获取的数据量
samples_per_epoch = 64

# 创建自定义采样器
indices = torch.randperm(len(train_data))[:samples_per_epoch]
sampler = SubsetSampler(indices)


train_loader = DataLoader(dataset=train_data, batch_size=16, sampler=sampler)

    
for epoch in range(1): 
  
    for images in train_loader:
        images = images.permute(0, 2, 3, 1) 
        # show_images(images)
        break

   