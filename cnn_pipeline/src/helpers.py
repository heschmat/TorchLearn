import os
import random
import requests
from zipfile import ZipFile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import multiprocessing

from tqdm import tqdm


def setup_env(url, directory= 'data'):
    if torch.cuda.is_available():
        print('GPU available!')
    else:
        print('GPU *NOT* available. Will use CPU (slow computation)')

    # Seed random generators
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    download_and_extract_data(url, directory)
    compute_mean_and_std(directory + '/train', batch_size= 64)

    # Make `checkpoints` subdir if not existing:
    os.makedirs('checkpoints', exist_ok= True)


def download_and_extract_data(url, directory):
    try:
        # Check if the directory exists and is empty
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Only proceed if the directory is empty
        if not os.listdir(directory):
            # Download the url data as `data.zip` in `directory`
            response = requests.get(url, stream= True)
            zip_path = os.path.join(directory, 'data.zip')
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size= 128):
                    f.write(chunk)
            # Extract `data.zip` and remove it afterwards:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(directory)
            os.remove(zip_path)

            print(f"Data downloaded and extracted to '{directory}'.")
        else:
            print(
                f"Directory '{directory}' already exists and is not empty."
                "To re-download, please remove the directory first."
            )
    except OSError as e:
        print(f"An error occurred: {e}")


def compute_mean_and_std(data_loc, batch_size= 64):
    """
    Compute per-channel mean and std of the dataset.
    Will be used to normalize images via transforms.Normalize().

    Parameters
    ----------
    data_loc: str
    batch_size: int

    Returns:
    (mean & std): (float, float)
    -------
    """
    cache_file = 'mean_std.pt'
    if os.path.exists(cache_file):
        print(f'Reusing the already available mean and std.')
        stats = torch.load(cache_file)
        return stats['mean'], stats['std']

    if batch_size == 1:
        print('No resizing will be done; but the computation will be slower!')
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
    ])
    ds = datasets.ImageFolder(
        data_loc, transform= transform
    )
    dl = DataLoader(ds, batch_size= batch_size, num_workers= multiprocessing.cpu_count())

    # Accumulate sums and squared sums
    mean = torch.zeros(3)
    var = torch.zeros(3)
    total_pixels = 0

    for images, _ in tqdm(dl, desc='Computing MEAN and STD', ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # Flatten H and W into one dimension
        mean += images.mean(dim=[0, 2]) * batch_samples
        var += images.var(dim=[0, 2], unbiased=False) * batch_samples
        total_pixels += batch_samples

    mean /= total_pixels
    std = torch.sqrt(var / total_pixels)

    # Save for future use:
    torch.save({'mean': mean, 'std': std}, cache_file)

    return mean, std

