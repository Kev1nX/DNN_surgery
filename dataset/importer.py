import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10_dataset():
    def __init__(self, batch_size=4, num_workers=2):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                shuffle=True, num_workers=self.num_workers)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size,
                                                shuffle=False, num_workers=self.num_workers)

        self.classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')