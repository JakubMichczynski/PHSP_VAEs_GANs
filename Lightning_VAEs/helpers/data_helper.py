import torch
# from torch.utils.data.sampler import SubsetRandomSampler
# from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from typing import List, Tuple, Type

import numpy as np
import pandas as pd
import copy
import random


class PhotonsDataset(Dataset):
    def __init__(self, data, transform=None):
        self.x = data
        self.n_samples = len(data)
        self.transform = transform
        self.columns = ['X', 'Y', 'dX', 'dY', 'dZ', 'E']

    def __getitem__(self, index):
        sample = self.x

        if self.transform:
            sample = self.transform(sample)
        return sample[index]

    def __len__(self):
        return self.n_samples


class ToTensorFromNdarray:
    def __call__(self, sample):
        sample
        return torch.from_numpy(sample)


def get_standarized_constrains(constrains_list_min: List[float], constrains_list_max: List[float], stdcs: Type[StandardScaler], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns tensors created on a device with constraining parameters standardized with the help of a StandardScaler object
    constrains_min = np.asarray(
        constrains_list_min, dtype=np.float32).reshape(1, -1)
    constrains_max = np.asarray(
        constrains_list_max, dtype=np.float32).reshape(1, -1)
    standarized_constrains_min = torch.tensor(
        stdcs.transform(constrains_min), device=device)
    standarized_constrains_max = torch.tensor(
        stdcs.transform(constrains_max), device=device)
    return standarized_constrains_min, standarized_constrains_max


def get_photons_with_introduced_XY_symmetries(photons: np.ndarray, random_seed: int) -> np.ndarray:
    # Returns photons with entered X symmetry and Y symmetry depending on random_seed
    random.seed(random_seed)
    symetrized_photons = copy.deepcopy(photons)
    for photon in symetrized_photons:
        if random.uniform(0, 1) > 0.5:
            photon[1] = -photon[1]
            photon[2] = -photon[2]
            photon[3] = -photon[3]
            photon[4] = -photon[4]
    return symetrized_photons


def get_train_validation_test_photonsdataloaders_and_standarscaler_from_numpy(photons: np.ndarray, batch_size: int, test_fraction: float = 0.2, validation_fraction: float = 0.2, train_transforms=None, test_transforms=None, num_workers: int = 0, random_seed: int = 123):
    # It splits the data set into a test, training and validation set, then standardizes the data based on the training set and returns the associated Dataloader for each member
    if train_transforms is None:
        train_transforms = ToTensorFromNdarray()

    if test_transforms is None:
        test_transforms = ToTensorFromNdarray()

    if test_fraction != 0:
        X_train_and_validation, X_test = train_test_split(
            photons, test_size=test_fraction, random_state=random_seed, shuffle=True)
    else:
        X_train_and_validation = copy.deepcopy(photons)

    if validation_fraction != 0:
        X_train, X_validation = train_test_split(
            X_train_and_validation, test_size=validation_fraction/(1-test_fraction), random_state=random_seed, shuffle=True)
    else:
        X_train = X_train_and_validation

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    train_dataset = PhotonsDataset(
        data=X_train_std, transform=train_transforms)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True)

    if test_fraction != 0:
        # wykorzystujemy standaryzacje z danych treningowych
        X_test_std = stdsc.transform(X_test)
        test_dataset = PhotonsDataset(
            data=X_test_std, transform=test_transforms)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True)
    else:
        test_loader = None

    if validation_fraction != 0:
        # wykorzystujemy standaryzacje z danych treningowych
        X_validation_std = stdsc.transform(X_validation)
        valid_dataset = PhotonsDataset(
            data=X_validation_std, transform=test_transforms)
        validation_loader = DataLoader(dataset=valid_dataset,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=True)
    else:
        validation_loader = None

    return train_loader, validation_loader, test_loader, stdsc


def get_train_validation_test_photonsDatasets_and_standarscaler_from_numpy(photons: np.ndarray, test_fraction: float = 0.2, validation_fraction: float = 0.0, train_transforms=None, test_transforms=None, random_seed: int = 123):
    # It splits the data set into a test, training and validation set, then standardizes the data based on the training set and returns the associated Dataset for each member
    if train_transforms is None:
        train_transforms = ToTensorFromNdarray()

    if test_transforms is None:
        test_transforms = ToTensorFromNdarray()

    if test_fraction != 0:
        X_train_and_validation, X_test = train_test_split(
            photons, test_size=test_fraction, random_state=random_seed, shuffle=True)
    else:
        X_train_and_validation = copy.deepcopy(photons)

    if validation_fraction != 0:
        X_train, X_validation = train_test_split(
            X_train_and_validation, test_size=validation_fraction/(1-test_fraction), random_state=random_seed, shuffle=True)
    else:
        X_train = X_train_and_validation

    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    train_dataset = PhotonsDataset(
        data=X_train_std, transform=train_transforms)

    if test_fraction != 0:
        # wykorzystujemy standaryzacje z danych treningowych
        X_test_std = stdsc.transform(X_test)
        test_dataset = PhotonsDataset(
            data=X_test_std, transform=test_transforms)
    else:
        test_dataset = None

    if validation_fraction != 0:
        # wykorzystujemy standaryzacje z danych treningowych
        X_validation_std = stdsc.transform(X_validation)
        valid_dataset = PhotonsDataset(
            data=X_validation_std, transform=test_transforms)
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, test_dataset, stdsc
