'''
This class is used to load the frames from the video and return the frames as a tensor.
'''


import os
from torch.utils.data import Dataset
import natsort
from PIL import Image

class LoadFrames(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        


    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        
        return tensor_image


