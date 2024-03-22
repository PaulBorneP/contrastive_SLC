# adapted from https://github.com/Yusin2Chen/self-supervised_change_detetction Copyright (c) 2021 Yusin Chen

import torch
from typing import List
import torch


class HardNegLoss:
    def __init__(self, batch_size: int, device: str, temperature: float = 0.5):
        """
        Initialize the HardNegLoss class.

        Args:
            batch_size (int): The batch size.
            device (str): The device to use for computations.
            temperature (float, optional): The temperature parameter for the loss calculation. Defaults to 0.5.
        """
        self.batch_size = batch_size
        self.device = device
        self.temperature = temperature

    def get_negative_mask(self) -> torch.Tensor:
        """
        Generate a mask to select negative samples. The mask is a batch_size x 2 * batch_size matrix.
            We want to select each patch once as a positive sample and the concatenation of the real features, and the imaginary part `out = torch.cat([out_1, out_2], dim=0)`is of size 2 * batch_size x feature_size.
            The negative mask as ones for the index of the real and imaginary part of the positive sample.
        Returns:
            torch.Tensor: The negative mask.
        """
        negative_mask = torch.ones(
            (self.batch_size, 2 * self.batch_size), dtype=bool)
        for i in range(self.batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + self.batch_size] = 0
        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def hard_neg_loss(self, out_1: torch.Tensor, out_2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the hard negative loss for the real and imaginary feature vectors of a batch of patches

        Args:
            out_1 (torch.Tensor): The output features of real part of the patches (batch_size x feature_size).
            out_2 (torch.Tensor): The output features of imaginary part of the patches 

        Returns:
            torch.Tensor: The calculated loss.
        """
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = self.get_negative_mask().to(self.device)
        neg = neg.masked_select(mask).view(2 * self.batch_size, -1)
        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        Ng = neg.sum(dim=-1)
        # contrastive loss
        loss = (- torch.log(pos / (pos + Ng))).mean()
        return loss

    def __call__(self, out_1: torch.Tensor, out_2: torch.Tensor) -> torch.Tensor:
        """
        Call the HardNegLoss instance.

        Args:
            out_1 (torch.Tensor): The output tensor 1.
            out_2 (torch.Tensor): The output tensor 2.

        Returns:
            torch.Tensor: The calculated loss.
        """
        return self.hard_neg_loss(out_1, out_2)
