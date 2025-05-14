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
        img = pad(img)
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

def pad(img, target_size=512):
    height, width = img.shape[:2]

    square_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    ratio = min(target_size / width, target_size / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    resized_img = cv2.resize(img, (new_width, new_height))

    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    square_img[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        resized_img
    )

    return square_img
