import torch

class HardNegLoss:
    def __init__(self, batch_size, device, temperature=0.5):
        self.batch_size = batch_size
        self.device = device
        self.temperature = temperature

    def get_negative_mask(self):
        negative_mask = torch.ones((self.batch_size, 2 * self.batch_size), dtype=bool)
        for i in range(self.batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + self.batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def hard_neg_loss(self, out_1, out_2):
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

    def __call__(self, out_1, out_2):
        return self.hard_neg_loss(out_1, out_2)