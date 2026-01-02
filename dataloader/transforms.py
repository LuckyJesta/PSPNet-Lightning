import torch
import random
import math
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

class SegmentationTransform:
    def __init__(self, size=(320, 320), is_train=True):
        self.size = size
        self.is_train = is_train

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image, mask):
        w, h = image.size
        tw, th = self.size
        scale = max(tw/w, th/h)

        new_h = int(math.ceil(h * scale))
        new_w = int(math.ceil(w * scale))
        new_h = max(new_h, th)
        new_w = max(new_w, tw)

        image = TF.resize(image, (new_h, new_w),
                          interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (new_h, new_w),
                         interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        else:
            image = TF.center_crop(image, self.size)
            mask = TF.center_crop(mask, self.size)


        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size}, is_train={self.is_train})'
