import os
import numpy as np
# import scipy.io as sio
from pathlib import Path
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.data import *

__all__ = ['CTW2019DataLoader', 'PreFetcher']


class PreFetcher:
    r""" Data pre-fetcher to accelerate the data loading
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.len = len(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None

    def preload(self):
        try:
            self.next_input = next(self.loader)
        except StopIteration:
            self.next_input = None
            return

        with torch.cuda.stream(self.stream):
            for idx, tensor in enumerate(self.next_input):
                self.next_input[idx] = tensor.cuda(non_blocking=True)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.loader = iter(self.ori_loader)
        self.preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        if input is None:
            raise StopIteration
        for tensor in input:
            tensor.record_stream(torch.cuda.current_stream())
        self.preload()
        return input
    

class CTW2019DataLoader(object):
    r""" PyTorch DataLoader for CTW2019 dataset.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory, scenario):
        assert os.path.isdir(root)
        root = Path(root)
        assert scenario in {"random", "wide", "narrow", "whthin"}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        channel, nt, nc = 2, 16, 924
        
        # load data
        # if (scenario == "random"):
        #     data = random_split_dataset(dataset_dir=root, split=0.9, shuffle=True, memory=None, verbose=True, random_state=None)
        # elif (scenario == "wide"):
        #     data = long_edge_section_split_dataset(dataset_dir=root, shuffle=True, memory=None, verbose=True, random_state=None)
        # elif (scenario == "narrow"):
        #     data = short_edge_section_split_dataset(dataset_dir=root, shuffle=True, memory=None, verbose=True, random_state=None)
        # elif (scenario == "whthin"):
        #     data = within_area_split_dataset(dataset_dir=root, shuffle=True, memory=None, verbose=True, random_state=None)

        # x_train, x_test, y_train, y_test = data[0][0],data[1][0],data[0][1],data[1][1]
        data_root = os.path.join(root, f"CTW2019/{scenario}.h5")
        with h5py.File(data_root, 'r') as f:
            x_train = f['h_train'][:]
            x_test = f['h_test'][:]
            y_train = f['pos_train'][:]
            y_test = f['pos_test'][:]

        assert x_train.shape[-3:] == (nt, nc, channel)
        h_train = torch.tensor(np.transpose(x_train, (0, 3, 1, 2)))
        h_test = torch.tensor(np.transpose(x_test, (0, 3, 1, 2)))
        pos_train = torch.tensor(y_train)
        pos_test = torch.tensor(y_test)

        self.train_dataset = TensorDataset(h_train, pos_train)
        self.test_dataset = TensorDataset(h_test, pos_test)

    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True,
                                  drop_last=True)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=32,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False,
                                 drop_last=True)

        # Accelerate CUDA data loading with pre-fetcher if GPU is used.
        if self.pin_memory is True:
            train_loader = PreFetcher(train_loader)
            # val_loader = PreFetcher(val_loader)
            test_loader = PreFetcher(test_loader)

        return train_loader, test_loader


        