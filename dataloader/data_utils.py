from torch.utils.data import DataLoader
from dataloader.dataset import LazyPetDataset
from dataloader.transforms import SegmentationTransform

def get_dataloaders(cfg):
    train_tf = SegmentationTransform(size=(320, 320), is_train=True)
    val_tf = SegmentationTransform(size=(320, 320), is_train=False)

    train_subset = LazyPetDataset(cfg.dataset.data_path.train, transform=train_tf)
    val_subset = LazyPetDataset(cfg.dataset.data_path.val, transform=val_tf)
    test_subset = LazyPetDataset(cfg.dataset.data_path.test, transform=val_tf)

    train_loader = DataLoader(train_subset, batch_size=cfg.dataset.batch_size, 
                              shuffle=True, num_workers=cfg.dataset.num_workers)
    val_loader = DataLoader(val_subset, batch_size=cfg.dataset.batch_size, 
                            shuffle=False, num_workers=cfg.dataset.num_workers)
    test_loader = DataLoader(test_subset, batch_size=cfg.dataset.batch_size, 
                             shuffle=False, num_workers=cfg.dataset.num_workers)
    
    return train_loader, val_loader, test_loader