import os
import torch
import numpy as np
from PIL import Image
import pytorch_lightning as L
from pytorch_lightning.callbacks import Callback

class ImageVisualizationCallback(Callback):
    def __init__(self, save_dir, num_samples=2):
        super().__init__()
        self.save_dir = os.path.join(save_dir, "vis_results")
        self.num_samples = num_samples
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        os.makedirs(self.save_dir, exist_ok=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        imgs, masks = batch
        preds=outputs["preds"]
        limit = min(len(imgs), self.num_samples)
        
        for i in range(limit):
            save_name = os.path.join(self.save_dir, f"batch{batch_idx}_img{i}.jpg")
            self._save_image(imgs[i], preds[i], save_name, pl_module.device)

    def _save_image(self, img_tensor, mask_tensor, save_path, device):
        mean = self.mean.to(device)
        std = self.std.to(device) 

        img = img_tensor * std + mean
        img = torch.clamp(img, 0, 1)
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        original_img = Image.fromarray(img_np).convert("RGBA")

        mask_np = mask_tensor.cpu().numpy()
        h, w = mask_np.shape
        color_mask = np.zeros((h, w, 4), dtype=np.uint8)
        
        color_mask[mask_np == 1] = [255, 0, 0, 100]
        color_mask[mask_np == 2] = [255, 255, 0, 100]

        mask_img = Image.fromarray(color_mask, mode="RGBA")

        final_img = Image.alpha_composite(original_img, mask_img)
        final_img.convert("RGB").save(save_path)