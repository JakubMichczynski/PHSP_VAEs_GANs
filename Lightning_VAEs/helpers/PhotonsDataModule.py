import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import numpy as np

from .data_helper import get_train_validation_test_photonsDatasets_and_standarscaler_from_numpy, get_photons_with_introduced_XY_symmetries

class PhotonsWithConditionalsDataModule(pl.LightningDataModule):
    def __init__(self, data_path=None, batch_size=None, transfrom=None, train_transform=None, test_transform=None, num_workers=0, test_fraction=0.0, validation_fraction=0.0, shuffle_train=True, random_seed=123, columns_keys=None):
        super().__init__()
        self.data_path = data_path
        self.columns_keys = columns_keys
        self.my_train_transform = train_transform
        self.my_test_transform = test_transform
        self.transform = transfrom
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.shuffle_train = shuffle_train
        self.random_seed = random_seed
        self.save_hyperparameters()
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def setup(self, stage: Optional[str] = None):
        # ODCZYTANIE DANYCH Z PLIKU 'photons.npy'
        photons_full = np.load(self.data_path)
        # ZMIANA NA FLOAT32
        photons_full = photons_full.astype(np.float32)
        # USUWANIE DANYCH Z dZ MNIEJSZYM NIŻ 0
        photons_full = np.delete(photons_full, np.where(
            photons_full[:, 5] < 0), axis=0)
        # USUWANIE PARAMETRÓW WARUNKOWYCH
        photons_full = photons_full[:,:6]

        self.photons_train, self.photons_val, self.photons_test, self.stdcs = get_train_validation_test_photonsDatasets_and_standarscaler_from_numpy(
            photons=photons_full, test_fraction=self.test_fraction, validation_fraction=self.validation_fraction, train_transforms=self.my_train_transform, test_transforms=self.my_test_transform, random_seed=self.random_seed)

    def train_dataloader(self):
        return DataLoader(dataset=self.photons_train, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.photons_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.photons_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

class PhotonsDataModule(pl.LightningDataModule):
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
        # #WPROWADZENIE SYMETRII X,Y DO ZBIORU FOTONÓW
        photons_full=get_photons_with_introduced_XY_symmetries(photons=photons_full, random_seed=self.random_seed)

        self.photons_train, self.photons_val, self.photons_test, self.stdcs = get_train_validation_test_photonsDatasets_and_standarscaler_from_numpy(
            photons=photons_full, test_fraction=self.test_fraction, validation_fraction=self.validation_fraction, train_transforms=self.my_train_transform, test_transforms=self.my_test_transform, random_seed=self.random_seed)
        

    def train_dataloader(self):
        return DataLoader(dataset=self.photons_train, batch_size=self.batch_size, shuffle=self.shuffle_train, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.photons_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.photons_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)
