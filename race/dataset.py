import tarfile
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import scipy.io
import os
import h5py

cache_path = "imdb_cache.h5"


class IMDBDataset(Dataset):
    def __init__(self, tar_path, limit=999999999):
        with tarfile.open(tar_path, "r") as tar:
            mat_data = scipy.io.loadmat(tar.extractfile("imdb_crop/imdb.mat"))
            num_samples = len(mat_data["imdb"][0][0][0][0])

            self.num_samples = min(limit, num_samples)

            if not (
                os.path.exists(cache_path)
                and self.num_samples <= (h5 := h5py.File(cache_path, "r")).attrs["num_samples"]
            ):
                self._create_cache(tar, mat_data)
            else:
                h5.close()

            self.h5 = h5py.File(cache_path, "r")

    def _create_cache(self, tar, mat_data):
        with h5py.File(cache_path, "w") as h5:
            h5.attrs["num_samples"] = self.num_samples
            images = h5.create_dataset(
                "images", shape=(self.num_samples, 512, 512, 3), dtype=np.uint8
            )
            ages = h5.create_dataset("ages", shape=(self.num_samples,), dtype=np.int32)
            celeb_ids = h5.create_dataset("celeb_ids", shape=(self.num_samples,), dtype=np.int32)

            for i in range(self.num_samples):
                dob = mat_data["imdb"][0][0][0][0][i]
                photo_taken = mat_data["imdb"][0][0][1][0][i]
                img_path = mat_data["imdb"][0][0][2][0][i][0]
                celeb_id = mat_data["imdb"][0][0][9][0][i]

                images[i] = pad(
                    cv2.imdecode(
                        np.frombuffer(
                            tar.extractfile("imdb_crop/" + img_path).read(), np.uint8
                        ),
                        cv2.IMREAD_ANYCOLOR,
                    )
                )
                ages[i] = photo_taken - dob // 365.25
                celeb_ids[i] = celeb_id

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.h5["images"][idx]
        age = self.h5["ages"][idx]
        celeb_id = self.h5["celeb_ids"][idx]

        return (
            torch.from_numpy(img).permute(2, 0, 1).float(),
            torch.tensor(age, dtype=torch.int),
            torch.tensor(celeb_id, dtype=torch.int),
        )

    def __del__(self):
        self.h5.close()


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
