from torchvision.datasets import CIFAR100
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import numpy as np
import scipy.io
import os

class IMDBDataset(Dataset):
    def __init__(self, mat_file, root_dir, limit=None):
        mat_data = scipy.io.loadmat(mat_file)
        num_samples = len(mat_data["imdb"][0][0][0][0])

        self.root_dir = root_dir
        self.num_samples = num_samples if limit == None else limit

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        self.ages = np.zeros(self.num_samples, dtype=int)
        self.img_paths = [None] * self.num_samples
        self.celeb_ids = np.zeros(self.num_samples, dtype=int)

        for i in range(self.num_samples):
            dob = mat_data["imdb"][0][0][0][0][i]
            photo_taken = mat_data["imdb"][0][0][1][0][i]
            self.img_paths[i] = mat_data["imdb"][0][0][2][0][i][0]
            self.celeb_ids[i] = mat_data["imdb"][0][0][9][0][i]
            
            self.ages[i] = photo_taken - dob // 365.25

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.img_paths[idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize_with_pad(img, (224, 224), (0, 0, 0))
        img = img.astype(np.float32) / 255.0

        if img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        
        age_class = age_to_class(self.ages[idx])
        
        return img, age_class
        
def age_to_class(age):
    if age <= 13:
        return 0
    elif age <= 26:
        return 1
    elif age <= 39:
        return 2
    elif age <= 52:
        return 3
    else:
        return 4

def resize_with_pad(image, new_shape, padding_color):
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_data[index][0]
            y = 1
            return x,y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x,y