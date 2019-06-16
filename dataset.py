# -*- coding: utf-8 -*-



from torch.utils.data import Dataset
import os
from PIL import Image

class FlatFolderDataset(Dataset):
    def __init__(self, img_dir, transform):  # path: input/content
        super(FlatFolderDataset, self).__init__()
        
        self.img_dir = img_dir
        self.imgs_name = os.listdir(img_dir)
        self.transform = transform
    
    def __getitem__(self, index):
        img_name = self.imgs_name[index]
        path = os.path.join(self.img_dir, img_name)
        img = Image.open(path).convert('RGB')
        img = self.transform(img)  # N x 3 x 256 x 256
        return {'img': img, 'img_name': img_name}

    def __len__(self):
        return len(self.imgs_name)


