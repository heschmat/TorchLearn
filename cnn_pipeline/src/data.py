
import os
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision import datasets, transforms

import multiprocessing
from tqdm import tqdm


def get_data_loaders(directory, batch_size, valid_size, num_workers= 1, limit= 0):
    """
    directory: str
    limit: maximum number of datapoints to consider
    """
    directory = Path(directory)
    if num_workers == -1:
        num_workers = multiprocessing.cpu_count()

    data_loaders = {'train': None, 'valid': None, 'test': None}

    mean, std = compute_mean_and_std(directory, 16)
    print(f'Dataset mean: {mean}, std: {std}')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            #transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean= mean, std= std)
        ]),

        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean= mean, std= std)
        ])
    }
    data_transforms['test'] = data_transforms['valid']

    # create train & validation datasets:
    trn_data = datasets.ImageFolder(
        directory / 'train', transform= data_transforms['train']
    )
    dev_data = datasets.ImageFolder(
        directory / 'train', transform= data_transforms['valid']
    )

    # Per request we could *limit* the size of datasets using `limit`
    # useful when debugging, or experimenting, or due to limited resources.
    n_images = limit if limit > 0 else len(trn_data)
    indices_rnd = torch.randperm(n_images)

    split_idx = int(math.ceil(valid_size * n_images))
    dev_idx, trn_idx = indices_rnd[:split_idx], indices_rnd[split_idx:]

    trn_sampler = SubsetRandomSampler(trn_idx)
    dev_sampler = SubsetRandomSampler(dev_idx)

    params_ = {'batch_size': batch_size, 'num_workers': num_workers}
    data_loaders['train'] = DataLoader(trn_data, sampler= trn_sampler, **params_)
    data_loaders['valid'] = DataLoader(dev_data, sampler= dev_sampler, **params_)

    # Create the test DataLoader:
    tst_data = datasets.ImageFolder(
        directory / 'test', transform= data_transforms['test']
    )

    if limit <= 0:
        tst_sampler = None
    else:
        # No need to shuffle test data: hence `arange` and not `randperm`.
        tst_sampler = SubsetRandomSampler(torch.arange(limit))

    data_loaders['test'] = DataLoader(
        tst_data,
        sampler= tst_sampler,
        shuffle= False,
        **params_
    )

    return data_loaders

