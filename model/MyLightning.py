import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as L
from torch.optim.lr_scheduler import PolynomialLR, StepLR
from model.model import PSPNet
import torchmetrics

class MyLightningModule(L.LightningModule):
    def __init__(self, model_cfg, optim_cfg, sched_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.optim_cfg = optim_cfg
        self.sched_cfg = sched_cfg
        self.num_classes = model_cfg.settings.num_classes
        
        self.model = PSPNet(**model_cfg.settings)
        
        self.example_input_array = torch.zeros(1, 3, 320, 320)

        self.criterion = nn.CrossEntropyLoss()

        self.test_miou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=self.num_classes,
            average="macro"
        )

        self.test_acc= torchmetrics.Accuracy(
            task="multiclass",
            num_classes=self.num_classes
        )


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks = batch
        
        output = self.model(imgs)
        loss = self.criterion(output, masks)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss 

    def validation_step(self, batch, batch_idx):
        imgs, masks = batch
        output = self.model(imgs)
        loss = self.criterion(output, masks)
        
        pred_mask = torch.argmax(output, dim=1)
        
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return {"val_loss": loss, "preds": pred_mask, "targets": masks}

    def test_step(self, batch, batch_idx):
        imgs, masks = batch
        output = self.model(imgs)
        pred_mask = torch.argmax(output, dim=1)
        
        loss = self.criterion(output, masks)
        self.test_miou(pred_mask, masks)
        self.test_acc(pred_mask, masks)
        
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_miou", self.test_miou, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True, prog_bar=True)

        return {"test_loss": loss, "preds": pred_mask, "targets": masks}

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(), 
            **self.optim_cfg
            )

        sched_name=self.sched_cfg.get("scheduler_name", "step")

        logging.info(f"Using {sched_name} learning rate scheduler.")

        if sched_name == "poly":
            total_steps = self.trainer.estimated_stepping_batches

            scheduler = PolynomialLR(
                optimizer,
                total_iters=total_steps,
                power=self.sched_cfg.power
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  
                    "frequency": 1
                }
            }

        elif sched_name == "step":
            scheduler = StepLR(
                optimizer,
                step_size=self.sched_cfg.get("step_size", 10),
                gamma=self.sched_cfg.get("gamma", 0.1)
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
                "frequency": 1
                }
            }