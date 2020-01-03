from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from .dataset import Datasets


class CIFAR10(Datasets):
    def __init__(self, root):
        super(CIFAR10, self).__init__(root=root)
        self.mean = [0.49139968, 0.48215827, 0.44653124]
        self.std = [0.24703233, 0.24348505, 0.26158768]

        self.dataset = datasets.CIFAR10
        self.num_classes = 10

    def get_loader(self, conf):
        normTransform = transforms.Normalize(self.mean, self.std)

        # post_process = [transforms.ToTensor(), normTransform]

        trainTransform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normTransform
        ])

        testTransform = transforms.Compose([
            transforms.ToTensor(),
            normTransform
        ])

        kwargs = {'num_workers': conf.workers, 'pin_memory': True} if conf.gpus > 0 else {}

        train_loader = DataLoader(
            self.dataset(root=conf.data, train=True, download=True,
                     transform=trainTransform),
            batch_size=conf.batch_size, shuffle=True, **kwargs)

        valid_loader = DataLoader(
            self.dataset(root=conf.data, train=False, download=True,
                     transform=testTransform),
            batch_size=conf.batch_size, shuffle=False, **kwargs)

        return train_loader, valid_loader


class CIFAR100(Datasets):
    def __init__(self, root):
        super(CIFAR100, self).__init__(root=root)
        self.mean = [0.49139968, 0.48215827, 0.44653124]
        self.std = [0.24703233, 0.24348505, 0.26158768]

        self.dataset = datasets.CIFAR100
        self.num_classes = 100

