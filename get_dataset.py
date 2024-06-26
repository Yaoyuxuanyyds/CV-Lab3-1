from torchvision.transforms import transforms
from transforms import GaussianBlur
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from view import ViewGen
from torch.utils.data import DataLoader


class GetTransformedDataset:
    @staticmethod
    def get_simclr_transform(size, s=1):
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply(
                                                  [color_jitter], p=0.8),
                                              transforms.RandomGrayscale(
                                                  p=0.2),
                                              GaussianBlur(
                                                  kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_cifar10_train(self, n_views):
        return datasets.CIFAR10('/root/Lab3-1/data', train=True, 
                                transform=ViewGen(self.get_simclr_transform(64), n_views),
                                download=False)
    
    def get_cifar10_test(self):
        return datasets.CIFAR10('/root/Lab3-1/data', train=False,
                                transform=transforms.ToTensor(),
                                download=False)

    def get_tiny_imagenet_train(self, n_views):
        train_dir = '/root/Lab3-1/data/tiny-imagenet-200/train'
        return datasets.ImageFolder(train_dir, 
                                    transform=ViewGen(self.get_simclr_transform(64), n_views))
    
    def get_tiny_imagenet_test(self):
        val_dir = '/root/Lab3-1/data/tiny-imagenet-200/val'
        return datasets.ImageFolder(val_dir, 
                                    transform=transforms.Compose([
                                        transforms.Resize(64),
                                        transforms.ToTensor()
                                    ]))
    


def get_cifar100_data_loaders(shuffle=False, batch_size=256):
    train_dataset = datasets.CIFAR100('/root/Lab3-1/data', train=True, download=False,
                                    transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    
    test_dataset = datasets.CIFAR100('/root/Lab3-1/data', train=False, download=False,
                                    transform=transforms.ToTensor())

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            num_workers=2, drop_last=False, shuffle=shuffle)
    return train_loader, test_loader
