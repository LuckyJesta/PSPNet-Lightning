import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class LazyPetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.img_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "trimaps")

        self.img_list = [os.path.splitext(f)[0]
                         for f in os.listdir(self.img_dir)]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]

        img_path = os.path.join(self.img_dir, img_name + ".jpg")
        mask_path = os.path.join(self.mask_dir, img_name + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        mask_np = np.array(mask)
        mask_np = mask_np - 1

        mask = Image.fromarray(mask_np)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask
