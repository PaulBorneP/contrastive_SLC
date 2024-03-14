import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.losses import hard_neg_loss
from modules.networks import ResNet18



class Trainer:
    def __init__(self, args, online_network, target_network, optimizer, device):

        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.savepath = args.save_path
        self.max_epochs = args.epochs
        self.m = 0.996
        # self.n_classes = args.n_classes
        # self.patch_size = args.patch_size
        self.pbatch_size = args.pbatch_size
        self.train_pbatch_size = int(args.pbatch_size * 0.5)
        # self.unfold_stride = args.unfold_stride
        # self.val_patch_size = args.val_patch_size
        # self.val_unfold_stride = args.val_unfold_stride
        self.eval_freq = args.eval_freq
        self.save_freq = args.save_freq



    def train(self, train_loader, eval_loader):

        niter = 0

        for epoch_counter in range(self.max_epochs):
            train_loss = 0.0
            for idx, batch in enumerate(train_loader):
                # image = batch['image']
                # # split whole image to patches
                # patches = self.patchize(image, self.patch_size, self.unfold_stride)
                # P, C, pH, pW = patches.shape

                print(batch['image'].shape)

                patches = batch['image'].squeeze(0).to(self.device)
                P, C, pH, pW = patches.shape

                # random shuffle index
                shuffle_ids = torch.randperm(P).to(self.device)
                # shuffle for training
                this_patches = patches[shuffle_ids]
                # training in each pbatch_size
                quotient, remainder = divmod(P, self.train_pbatch_size)
                pbatch = quotient if quotient > 0 else remainder
                for i in range(pbatch):
                    start = i * self.train_pbatch_size
                    end = start + self.train_pbatch_size

                    patch = this_patches[start:end, :, :, :]
                    # read file
                    loss = self.update(patch)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    niter += 1
                    train_loss += loss.item()

                train_loss = train_loss / self.pbatch_size
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                    epoch_counter, train_loss))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if (epoch_counter + 1) % self.eval_freq == 0:
                self.validate(eval_loader)
                self.online_network.train()

            # save checkpoints
            if (epoch_counter + 1) % self.save_freq == 0:
                if not os.path.exists(self.savepath):
                    os.makedirs(self.savepath)
                self.save_model(os.path.join(self.savepath,
                                             'MIE_epoch_{epoch}_{loss}.pth'.format(epoch=epoch_counter, loss=train_loss)))

    def update(self, image):

        real_img = image[:, 0, ...].unsqueeze(1)
        imaginary_img = image[:, 1, ...].unsqueeze(1)

        r_feature = self.online_network(real_img)
        i_feature = self.target_network(imaginary_img)

        # check dimension
        r_feature = F.normalize(r_feature, dim=1)
        r_feature = F.normalize(r_feature, dim=1)

        loss = hard_neg_loss(r_feature, i_feature,
                                batch_size=self.train_pbatch_size,device=self.device)
        return loss

    # def patchize(self, img: torch.Tensor, patch_size, unfold_stride) -> torch.Tensor:

    #     """
    #     img.shape
    #     B  : batch size
    #     C  : channels of image (same to patches.shape[1])
    #     iH : height of image
    #     iW : width of image

    #     pH : height of patch
    #     pW : width of patch
    #     V  : values in a patch (pH * pW * C)
    #     """

    #     B, C, iH, iW = img.shape
    #     pH = patch_size
    #     pW = patch_size

    #     unfold = nn.Unfold(kernel_size=(pH, pW), stride=unfold_stride)

    #     patches = unfold(img)  # (B, V, P)
    #     patches = patches.permute(0, 2, 1).contiguous()  # (B, P, V)
    #     patches = patches.view(-1, C, pH, pW)  # (P, C, pH, pW)
    #     return patches


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
    

    online_network = ResNet18()
    target_network = ResNet18()

    optimizer = torch.optim.Adam(online_network.parameters(), lr=0.001)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # need to be bs=1 if we pass the patches directly
    train_loader = DataLoader(FakeDataset(), batch_size=1, shuffle=True)
    eval_loader = DataLoader(FakeDataset(), batch_size=1, shuffle=True)
    
    trainer = Trainer(args=args, online_network=online_network, target_network=target_network, optimizer=optimizer, device=device)

    trainer.train(train_loader, eval_loader)

    print('Training finished')

