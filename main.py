import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer:
    def __init__(self, args, online_network, target_network, optimizer, device):

        #DEFAULT_AUG = nn.Sequential(
        #    RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.5),
        #    RandomApply(geo.transform.Hflip(), p=0.5),
        #    RandomApply(geo.transform.Vflip(), p=0.5))
        #augment_fn = None
        #self.augment = default(augment_fn, DEFAULT_AUG)
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.savepath = args.save_path
        self.max_epochs = args.epochs
        self.m = 0.996
        self.n_classes = args.n_classes
        self.patch_size = args.patch_size
        self.pbatch_size = args.pbatch_size
        self.train_pbatch_size = int(args.pbatch_size * 0.5)
        self.unfold_stride = args.unfold_stride
        self.val_patch_size = args.val_patch_size
        self.val_unfold_stride = args.val_unfold_stride
        self.eval_freq = args.eval_freq
        self.save_freq = args.save_freq

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_loader, eval_loader):

        niter = 0

        for epoch_counter in range(self.max_epochs):
            train_loss = 0.0
            for idx, (batch, _) in enumerate(train_loader):
                image = batch['image']
                # split whole image to patches
                patches = self.patchize(image, self.patch_size, self.unfold_stride)
                P, C, pH, pW = patches.shape
                # random shuffle index
                shuffle_ids = torch.randperm(P).cuda()
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
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch_counter, train_loss))
                torch.cuda.empty_cache()

            if (epoch_counter + 1) % self.eval_freq == 0:
                self.validate(eval_loader)
                self.online_network.train()

            # save checkpoints
            if (epoch_counter + 1) % self.save_freq == 0:
                self.save_model(os.path.join(self.savepath,
                                             'MIE_epoch_{epoch}_{loss}.pth'.format(epoch=epoch_counter, loss=train_loss)))

    def update(self, image):
        # split pre and post
        batch_view_1, batch_view_2 = torch.split(image, [4, 4], dim=1)
        # if you want to train your network on single image
        #image1, _ = torch.split(image, [4, 4], dim=1)
        #batch_view_1, batch_view_2 = self.augment(image1), self.augment(image1)
        batch_view_1 = batch_view_1.to(self.device)
        batch_view_2 = batch_view_2.to(self.device)
        # compute query feature
        o_feature1 = self.online_network(batch_view_1)
        o_feature2 = self.online_network(batch_view_2)

        # compute key features
        #with torch.no_grad():
        #    t_feature2 = self.target_network(batch_view_1)
        #    t_feature1 = self.target_network(batch_view_2)

        # loss function options (normal contrastive loss)
        l_feature1 = F.normalize(o_feature1, dim=1)
        l_feature2 = F.normalize(o_feature2, dim=1)
        scores = torch.matmul(l_feature1, l_feature2.t())
        mi_estimation = smile_lower_bound(scores)
        loss = - mi_estimation
        # hard negative loss (it seems that using mean teacher in small data set leads to performance drops)
        #loss = HardNegtive_loss(o_feature1, t_feature1, batch_size=self.train_pbatch_size) +  \
        #       HardNegtive_loss(o_feature2, t_feature2, batch_size=self.train_pbatch_size)
        #loss = HardNegtive_loss(o_feature1, o_feature2, batch_size=self.train_pbatch_size)
        return loss
    
    def patchize(self, img: torch.Tensor, patch_size, unfold_stride) -> torch.Tensor:

        """
        img.shape
        B  : batch size
        C  : channels of image (same to patches.shape[1])
        iH : height of image
        iW : width of image

        pH : height of patch
        pW : width of patch
        V  : values in a patch (pH * pW * C)
        """

        B, C, iH, iW = img.shape
        pH = patch_size
        pW = patch_size

        unfold = nn.Unfold(kernel_size=(pH, pW), stride=unfold_stride)

        patches = unfold(img)  # (B, V, P)
        patches = patches.permute(0, 2, 1).contiguous()  # (B, P, V)
        patches = patches.view(-1, C, pH, pW)  # (P, C, pH, pW)
        return patches