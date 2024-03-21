import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb.wandb_run
from modules.losses import HardNegLoss

from typing import Any, Dict
import os
import torch

import wandb

class Trainer:
    def __init__(self, args: Any, online_network: nn.Module, target_network: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, logger:wandb.wandb_run.Run  = None):
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

        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.savepath = args.save_path
        self.max_epochs = args.epochs
        self.m = 0.996
        self.pbatch_size = args.pbatch_size
        self.train_pbatch_size = int(args.pbatch_size * 0.5)
        self.eval_freq = args.eval_freq
        self.save_freq = args.save_freq
        self.criterion = HardNegLoss(batch_size=self.train_pbatch_size, device=self.device)
        self.logger = logger

    def train(self, train_loader: Any, eval_loader: Any) -> None:
        """
        Train the model.

        Args:
            train_loader (Any): Training data loader.
            eval_loader (Any): Evaluation data loader.
        """
        niter = 0
        for epoch_counter in range(self.max_epochs):
            train_loss = 0.0
            for idx, batch in enumerate(train_loader):
                patches = batch['image'].squeeze(0).to(self.device)
                P, C, pH, pW = patches.shape
                shuffle_ids = torch.randperm(P).to(self.device)
                this_patches = patches[shuffle_ids]
                quotient, remainder = divmod(P, self.train_pbatch_size)
                pbatch = quotient if quotient > 0 else remainder
                for i in range(pbatch):
                    start = i * self.train_pbatch_size
                    end = start + self.train_pbatch_size
                    patch = this_patches[start:end, :, :, :]
                    loss = self.update(patch)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    niter += 1
                    train_loss += loss.item()
                train_loss = train_loss / self.pbatch_size
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch_counter, train_loss))
                self.logger.log({'train_loss': train_loss})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # if (epoch_counter + 1) % self.eval_freq == 0:
            #     self.validate(eval_loader)
            #     self.online_network.train()
            if (epoch_counter + 1) % self.save_freq == 0:
                if not os.path.exists(self.savepath):
                    os.makedirs(self.savepath)
                self.save_model(os.path.join(self.savepath, 'MIE_epoch_{epoch}_{loss}.pth'.format(epoch=epoch_counter, loss=train_loss)))

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
        

