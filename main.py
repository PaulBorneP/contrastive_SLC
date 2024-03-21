import os
import torch
from modules.networks import ResNet18
import wandb

from modules.trainer import Trainer


if __name__ == '__main__':
    # test on fake data
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Path to save checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--pbatch_size', type=int, default=4, help='Size of training patch batch')
    parser.add_argument('--eval_freq', type=int, default=1, help='Frequency of evaluation')
    parser.add_argument('--save_freq', type=int, default=1, help='Frequency of saving checkpoints')
    args = parser.parse_args()

    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset

    class FakeDataset(Dataset):
        def __init__(self):
            self.data = torch.rand(100, 10, 2, 64, 64)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return {'image': self.data[idx]}
    # create model
    
    logger = wandb.init(project='c_simp_comp')
    
    online_network = ResNet18()
    target_network = ResNet18()
    logger.watch(online_network)
    logger.watch(target_network)

    optimizer = torch.optim.Adam(online_network.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # need to be bs=1 if we pass the patches directly
    train_loader = DataLoader(FakeDataset(), batch_size=1, shuffle=True)
    eval_loader = DataLoader(FakeDataset(), batch_size=1, shuffle=True)
    
    trainer = Trainer(args=args, online_network=online_network, target_network=target_network, optimizer=optimizer, device=device, logger=logger)

    trainer.train(train_loader, eval_loader)

    print('Training finished')

