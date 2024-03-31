# import os
# import torch
# from modules.networks import ResNet18
# import wandb

# from modules.trainer import Trainer
# from database.dataloader import BatchMaker

# if __name__ == '__main__':
#     # test on fake data

#     patch_size =(24,24)
#     batch_size = 50
#     # deasable wandb
#     os.environ['WANDB_MODE'] = 'dryrun'
#     logger = wandb.init(project='c_simp_comp')
    
#     online_network = ResNet18()
#     target_network = ResNet18()
#     logger.watch(online_network)
#     logger.watch(target_network)

#     optimizer = torch.optim.Adam(online_network.parameters(), lr=0.001)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     train_loader = BatchMaker("/Users/newt/Desktop/MVA/RS/Project/contrastive_SLC/data/PileDomancyTSX26.IMA")
#     train_loader.init_area_ref(patch_size=patch_size)
#     train_loader.init_M0(patch_size=patch_size)

#     trainer = Trainer(online_network=online_network, target_network=target_network, optimizer=optimizer, batch_size=batch_size,patch_size=patch_size,device=device, logger=logger)

#     trainer.train(train_loader)

#     print('Training finished')


from logging import disable
import os
import torch
from modules.networks import ResNet18
import wandb
wandb.disabled = True
from modules.trainer import Trainer
from database.dataloader import BatchMaker
os.environ['WANDB_DISABLED'] = 'true'


patch_size =(16,16)
batch_size = 50
num_feats=128
online_network = ResNet18(num_feats=num_feats)
target_network = ResNet18(num_feats=num_feats)




optimizer = torch.optim.Adam(online_network.parameters(), lr=0.001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = BatchMaker("/Users/newt/Desktop/MVA/RS/Project/contrastive_SLC/data/PileDomancyTSX26.IMA")
train_loader.init_area_ref(patch_size=patch_size)
train_loader.init_M0(patch_size=patch_size)


config = {"patch_size":patch_size[0],"batch_size":batch_size,"network":online_network.__class__.__name__,"optimizer":optimizer.__class__.__name__,"lr":[pg['lr'] for pg in optimizer.param_groups][0],"num_feats":num_feats}
name = f"cs_comp_{config['network']}_in:{config['patch_size']}_out:{config['num_feats']}/"
save_path = os.path.join("/Users/newt/Desktop/MVA/RS/Project/contrastive_SLC/checkpoints/",name)
print(save_path)
if not os.path.exists(save_path):
  os.makedirs(save_path)

logger = wandb.init(project='c_simp_comp',name=name,config= config)
logger.watch(online_network)
logger.watch(target_network)

trainer = Trainer(online_network=online_network, target_network=target_network, optimizer=optimizer, batch_size=batch_size,patch_size=patch_size,device=device, logger=logger,save_path=save_path)





trainer.train(train_loader)



print('Training finished')

