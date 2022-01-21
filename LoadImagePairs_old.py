import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import natsort
from PIL import Image
import pdb

class LoadImagePairs(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = natsort.natsorted(os.listdir(main_dir))

        img_pairs = []
        for fn in range(0,len(all_imgs)-1):
            img_pairs.append([all_imgs[fn], all_imgs[fn+1]])

        self.total_imgs = natsort.natsorted(img_pairs)
        


    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc1 = os.path.join(self.main_dir, self.total_imgs[idx][0])
        img_loc2 = os.path.join(self.main_dir, self.total_imgs[idx][1])
        
        image1 = Image.open(img_loc1).convert("RGB")
        tensor_image1 = self.transform(image1)
        image2 = Image.open(img_loc2).convert("RGB")
        tensor_image2 = self.transform(image2)

        tensor_pair = torch.cat([tensor_image1, tensor_image2]).unsqueeze(0)
        
        return tensor_pair


""" transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])
hab_data = LoadFrames('Frames/Figure_23_Bulge',transform_)
trainloader = torch.utils.data.DataLoader(hab_data, batch_size=len(hab_data), shuffle=False, num_workers = 4, pin_memory=True)
#dataiter = iter(trainloader)
print(hab_data) """