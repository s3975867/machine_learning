from torchvision.datasets import CIFAR100
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import cv2
import numpy as np
import scipy.io
import os
import pandas as pd

class IMDBDataset(Dataset):
    def __init__(self, csv_file, root_dir, limit=None):
        df = pd.read_csv(csv_file)
        
        df = self._undersample_balanced(df, limit if limit else len(df))
        
        self.root_dir = root_dir
        self.num_samples = len(df)

        self.img_paths = df['img_path'].tolist()
        self.ages = df['age'].values
        self.celeb_ids = df['celeb_id'].values

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
    
    def _undersample_balanced(self, df, limit):
        df['age_class'] = df['age'].apply(age_to_class)
        
        celeb_counts = df.groupby('celeb_id').size()
        
        df = df.sort_values('celeb_id', key=lambda x: x.map(celeb_counts), ascending=False)
        
        class_sizes = df['age_class'].value_counts()
        minority_size = class_sizes.min()
        
        if limit is None:
            max_per_class = minority_size
        else:
            max_per_class = min(limit // 5, minority_size)
        
        result_df = df.groupby('age_class').head(max_per_class).reset_index(drop=True)
        
        print(f"Balanced dataset: {len(result_df)} samples across {len(result_df['age_class'].unique())} age classes")
        print(f"Unique celebrities: {result_df['celeb_id'].nunique()}")
        print("Class distribution:", result_df['age_class'].value_counts().sort_index().to_dict())
        
        return result_df

def get_celeb_name(celeb_id):
    mat_data = scipy.io.loadmat('./imdb_crop/imdb.mat')
    return mat_data["imdb"][0][0][8][0][celeb_id]

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