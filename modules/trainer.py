import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb.wandb_run
from modules.losses import HardNegLoss

from database.dataloader import BatchMaker

from typing import Any, Dict
import os
import torch

import wandb
from tqdm import tqdm


class Trainer:
    def __init__(self, online_network: nn.Module, target_network: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, temperature=0.5, n_iter: int = 30000, logger: wandb.wandb_run.Run = None, save_freq: int = 1000, batch_size: int = 50, patch_size=24, save_path: str = 'checkpoints'):
        """
        Initialize the Trainer class.

        Args:
            args (Any): Arguments for the trainer.
            online_network (nn.Module): The online network.
            target_network (nn.Module): The target network.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (torch.device): The device to run the training on.
            logger (Any, optional): Logger for logging training progress. Defaults to None.
        """

        self.online_network = online_network.to(device=device)
        self.target_network = target_network.to(device=device)
        self.online_network.train()
        self.target_network.train()
        self.optimizer = optimizer
        self.device = device
        self.savepath = save_path

        self.max_iter = n_iter
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.save_freq = save_freq
        self.criterion = HardNegLoss(
            batch_size=self.batch_size, device=self.device, temperature=temperature)
        self.logger = logger

    def train(self, train_batch_maker: BatchMaker) -> None:
        for i in tqdm(range(self.max_iter)):
            self.optimizer.zero_grad()
            batch, _ = train_batch_maker.make_batch_M0(
                self.batch_size, self.patch_size)
            batch = torch.tensor(batch, dtype=torch.float32).to(self.device)
            # check if double or float
            loss = self.update(batch)
            # log loss
            self.logger.log({'loss': loss.item()})
            loss.backward()
            self.optimizer.step()
            if i % self.save_freq == 0:
                torch.save(self.online_network.state_dict(),
                           os.path.join(self.savepath, f'online_{i}.pth'))
                torch.save(self.target_network.state_dict(),
                           os.path.join(self.savepath, f'target_{i}.pth'))
            if i % 100 == 0:
                tqdm.write(f'Loss at iteration {i}: {loss.item()}')

    def update(self, image: torch.Tensor) -> torch.Tensor:
        """
        Update the model.

        Args:
            image (torch.Tensor): Input image.

        Returns:
            torch.Tensor: Loss value.
        """
        real_img = image[:, 0, ...].unsqueeze(1)
        imaginary_img = image[:, 1, ...].unsqueeze(1)
        r_feature = self.online_network(real_img)
        i_feature = self.target_network(imaginary_img)
        r_feature = F.normalize(r_feature, dim=1)
        r_feature = F.normalize(r_feature, dim=1)
        loss = self.criterion(r_feature, i_feature)
        return loss
