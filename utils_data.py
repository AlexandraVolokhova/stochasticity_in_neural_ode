import numpy as np
import os
import torch
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler

DATA_ROOT = os.environ['DATA_ROOT']


def get_gauss2d_data(args):
    if args.data == "crossed_gaussians":
        data = make_blobs(n_samples=args.n_samples, n_features=2, centers=[args.mu_1, args.mu_2],
                          cluster_std=[args.std_1, args.std_2], random_state=args.data_seed)
    else:
        raise NotImplementedError
    return data


def get_gauss2d_loaders(args):
    x, y = get_gauss2d_data(args)
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=args.train_size, random_state=args.data_seed)
    test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.5, random_state=args.data_seed)
    trainset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x),
                                              torch.LongTensor(train_y))
    valset = torch.utils.data.TensorDataset(torch.FloatTensor(val_x),
                                            torch.LongTensor(val_y))
    testset = torch.utils.data.TensorDataset(torch.FloatTensor(test_x),
                                             torch.LongTensor(test_y))

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True,
                                              num_workers=1, pin_memory=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.val_bs, shuffle=True,
                                              num_workers=1, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False,
                                             num_workers=1, pin_memory=True, drop_last=False)
    return train_loader, val_loader, test_loader


def get_cifar100_loaders(batch_size=128, test_batch_size=1000, val_size=0.2, data_root=DATA_ROOT, limit=None,
                        verbose=False, augmentation=True):
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_root = os.path.join(data_root, 'cifar100')
    train_dataset = datasets.CIFAR100(
        root=data_root, train=True,
        download=True, transform=train_transform,
    )
    val_dataset = datasets.CIFAR100(
        root=data_root, train=True,
        download=True, transform=test_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=data_root, train=False,
        download=True, transform=test_transform,
    )

    num_train = len(train_dataset)
    if limit:
        num_train = limit
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if verbose:
        print("train size: {}\nsplit: {}\n".format(num_train, split))
    # random_seed = 30
    #     # np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, sampler=valid_sampler, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4
    )

    return train_loader, valid_loader, test_loader


def get_cifar10_loaders(batch_size=128, test_batch_size=1000, val_size=0.2, data_root=DATA_ROOT, limit=None,
                        verbose=False, augmentation=True):
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_root = os.path.join(data_root, 'cifar10')
    train_dataset = datasets.CIFAR10(
        root=data_root, train=True,
        download=True, transform=train_transform,
    )
    val_dataset = datasets.CIFAR10(
        root=data_root, train=True,
        download=True, transform=test_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False,
        download=True, transform=test_transform,
    )

    num_train = len(train_dataset)
    if limit:
        num_train = limit
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if verbose:
        print("train size: {}\nsplit: {}\n".format(num_train, split))
    # random_seed = 30
    #     # np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
    )
    valid_loader = DataLoader(
        val_dataset, batch_size=test_batch_size, sampler=valid_sampler, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4
    )

    return train_loader, valid_loader, test_loader
