import os
import torch
from modules.networks import ResNet18
import wandb

from modules.trainer import Trainer
from database.dataloader import BatchMaker

if __name__ == '__main__':
    # test on fake data

    patch_size =(24,24)
    batch_size = 50
    # deasable wandb
    os.environ['WANDB_MODE'] = 'dryrun'
    logger = wandb.init(project='c_simp_comp')
    
    online_network = ResNet18()
    target_network = ResNet18()
    logger.watch(online_network)
    logger.watch(target_network)

    optimizer = torch.optim.Adam(online_network.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = BatchMaker("/Users/newt/Desktop/MVA/RS/Project/contrastive_SLC/data/PileDomancyTSX26.IMA")
    train_loader.init_area_ref(patch_size=patch_size)
    train_loader.init_M0(patch_size=patch_size)

    trainer = Trainer(online_network=online_network, target_network=target_network, optimizer=optimizer, batch_size=batch_size,patch_size=patch_size,device=device, logger=logger)

    trainer.train(train_loader)

    print('Training finished')

