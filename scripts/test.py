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
from hydra.core.hydra_config import HydraConfig 
from dataloader import get_dataloaders
from model import MyLightningModule
from utils import ImageVisualizationCallback

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    
    L.seed_everything(42, workers=True)
    _, _, test_loader = get_dataloaders(cfg)
    ckpt_path = cfg.get("ckpt_path", None)
    
    if not ckpt_path:
        logging.warning("No 'ckpt_path' provided in config. Testing initialized weights.")
        model = MyLightningModule(cfg.model, cfg.optimizer, cfg.scheduler)
    else:
        logging.info(f"Testing with checkpoint: {ckpt_path}")

    model = MyLightningModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model_cfg=cfg.model, 
        optim_cfg=cfg.optimizer, 
        sched_cfg=cfg.scheduler
    )

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    logger = TensorBoardLogger(save_dir=output_dir, name="", version="")
    vis_callback= ImageVisualizationCallback(
        save_dir=output_dir,
        num_samples=cfg.visualization.num_samples
    )

    trainer = L.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=[vis_callback]
    )

    logging.info("Starting Testing ...")
    trainer.test(model, test_loader)
    logging.info("Testing Finished.")

if __name__ == "__main__":
    main()