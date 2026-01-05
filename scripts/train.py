import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
torch.set_float32_matmul_precision('medium') 

import hydra
from omegaconf import DictConfig
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from hydra.core.hydra_config import HydraConfig
from dataloader import get_dataloaders
from model import MyLightningModule, MyLightningModuleManual

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
        
    L.seed_everything(42, workers=True)

    train_loader, val_loader, _ = get_dataloaders(cfg)

    if cfg.manual_optimization:
        model = MyLightningModuleManual(
        model_cfg=cfg.model, 
        optim_cfg=cfg.optimizer,
        sched_cfg=cfg.scheduler
        )
        logging.info("Using manual optimization.")
    else:
        model = MyLightningModule(
        model_cfg=cfg.model, 
        optim_cfg=cfg.optimizer,
        sched_cfg=cfg.scheduler
        )
        logging.info("Using automatic optimization.")

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir 

    logger = TensorBoardLogger(save_dir=output_dir, name="", version="")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1,
        filename='best-{epoch:02d}-{val_loss:.2f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = L.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor]
    )

    logging.info(f"Starting Training ...")
    trainer.fit(model, train_loader, val_loader)
    logging.info("Training Finished.")

if __name__ == "__main__":
    main()