from argparse import Namespace
import torch
import torch.nn as nn
import torchvision.models as models
from get_dataset import GetTransformedDataset
from simclr import simclr_framework


class ResNet18(nn.Module):
    def __init__(self, out_dim):
        super(ResNet18, self).__init__()
        self.backbone = self._get_basemodel(out_dim)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, out_dim):
        return models.resnet18(weights=None, num_classes=out_dim)

    def forward(self, x):
        return self.backbone(x)
    

def main():
    args = Namespace
    # Hyperparameters
    args.batch_size = 512
    args.epochs = 100
    args.lr = 1e-4
    args.temperature = 0.07
    args.weight_decay = 0.001
    # Other settings
    args.fp16_precision = False
    args.device = torch.device('cuda')
    args.gpu_index = 0
    args.log_every_n_steps = 1
    args.n_views = 2
    args.out_dim = 128
    args.seed = 1
    args.workers = 4
    args.log_dir = f"/root/Lab3-1/logs/simclr_TIN_lr{args.lr}_wd{args.weight_decay}_bt{args.batch_size}_e{args.epochs}"

    print(args.device)

    dataset = GetTransformedDataset()
    train_dataset = dataset.get_cifar10_train(args.n_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNet18(out_dim=args.out_dim)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
 
    simclr = simclr_framework(model=model, optimizer=optimizer, scheduler=scheduler, args=args)

    print('training started..')
    simclr.train(train_loader)
    print('training completed..')

if __name__ == "__main__":
    main()