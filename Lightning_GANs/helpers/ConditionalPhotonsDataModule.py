import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np
import torch
import copy
import collections

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ToTensorFromNdarray:
    def __call__(self, sample):
        sample
        return torch.from_numpy(sample)

class ConditionalPhotonsDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.n_samples = len(data)
        self.transform = transform
        self.columns = ['X', 'Y', 'dX', 'dY', 'dZ', 'E', 'E_el']

    def __getitem__(self, index):
        sample = self.data[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

def get_train_validation_test_photonsDatasets_and_standarscaler_from_numpy(photons: np.ndarray, test_fraction: float = 0.2, validation_fraction: float = 0.0, train_transforms=None, test_transforms=None, random_seed: int = 123):
    # It splits the data set into a test, training and validation set, then standardizes the data based on the training set and returns the associated Dataset for each member
    if train_transforms is None:
        train_transforms = ToTensorFromNdarray()

    if test_transforms is None:
        test_transforms = ToTensorFromNdarray()

    if test_fraction != 0:
        X_train_and_validation, X_test = train_test_split(
            photons, test_size=test_fraction, random_state=random_seed, shuffle=True, stratify=photons[:,6])
    else:
        X_train_and_validation = copy.deepcopy(photons)

    if validation_fraction != 0:
        X_train, X_validation = train_test_split(
            X_train_and_validation, test_size=validation_fraction/(1-test_fraction), random_state=random_seed, shuffle=True, stratify=photons[:,6])
    else:
        X_train = X_train_and_validation

    stdsc = MinMaxScaler()
    X_train_std = stdsc.fit_transform(X_train)
    train_dataset = ConditionalPhotonsDataset(
        data=X_train_std, transform=train_transforms)

    if test_fraction != 0:
        # wykorzystujemy standaryzacje z danych treningowych
        X_test_std = stdsc.transform(X_test)
        test_dataset = ConditionalPhotonsDataset(
            data=X_test_std, transform=test_transforms)
    else:
        test_dataset = None

    if validation_fraction != 0:
        # wykorzystujemy standaryzacje z danych treningowych
        X_validation_std = stdsc.transform(X_validation)
        valid_dataset = ConditionalPhotonsDataset(
            data=X_validation_std, transform=test_transforms)
    else:
        valid_dataset = None

    return train_dataset, valid_dataset, test_dataset, stdsc


class ConditionalPhotonsDataModule(pl.LightningDataModule):
    def __init__(self, data_path=None, batch_size=None, transfrom=None, train_transform=None, test_transform=None, num_workers=0, test_fraction=0.0, validation_fraction=0.0, shuffle_train=True, random_seed=123, columns_keys=None):
        super().__init__()
        self.data_path = data_path
        self.columns_keys=columns_keys
        self.my_train_transform = train_transform
        self.my_test_transform = test_transform
        self.transform=transfrom
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.shuffle_train = shuffle_train
        self.random_seed = random_seed
        self.save_hyperparameters()
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


    def setup(self, stage: Optional[str] = None):
        #ODCZYTANIE DANYCH Z PLIKU 'photons.npy'
        photons_full = np.load(self.data_path)
        #USUWANIE DANYCH Z dZ MNIEJSZYM NIŻ 0
        photons_full=np.delete(photons_full, np.where(photons_full[:,5]<0),axis=0)

        self.photons_train, self.photons_val, self.photons_test, self.stdcs = get_train_validation_test_photonsDatasets_and_standarscaler_from_numpy(
            photons=photons_full, test_fraction=self.test_fraction, validation_fraction=self.validation_fraction, train_transforms=self.my_train_transform, test_transforms=self.my_test_transform, random_seed=self.random_seed)
        

    def train_dataloader(self):
        return DataLoader(dataset=self.photons_train, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.photons_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.photons_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
