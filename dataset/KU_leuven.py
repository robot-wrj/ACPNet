import os
import numpy as np
# import scipy.io as sio
from pathlib import Path
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

# from utils.data import *

__all__ = ['KUleuvenDataLoader', 'PreFetcher']


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
    

class KUleuvenDataLoader(object):
    r""" PyTorch DataLoader for CTW2019 dataset.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory, scenario, los="LoS", topology = "URA"):
        assert os.path.isdir(root)
        root = Path(root)
        assert scenario in {"random", "wide", "narrow", "whthin"}
        assert los in {"LoS", "nLoS"}
        assert topology in {"URA", "ULA", "DIS"}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        dim = (2, 64, 100)
        data_path = os.path.join(root, f"KU_leuven/ultra_dense/{topology}_lab_{los}")
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
        # data_root = os.path.join(root, f"CTW2019/{scenario}.h5")
        # with h5py.File(data_root, 'r') as f:
        #     x_train = f['h_train'][:]
        #     x_test = f['h_test'][:]
        #     y_train = f['pos_train'][:]
        #     y_test = f['pos_test'][:]

        # assert x_train.shape[-3:] == (nt, nc, channel)
        # h_train = torch.tensor(np.transpose(x_train, (0, 3, 1, 2)))
        # h_test = torch.tensor(np.transpose(x_test, (0, 3, 1, 2)))
        # # pos_train = torch.tensor(np.transpose(y_train, (0, 3, 1, 2)))
        # # pos_test = torch.tensor(np.transpose(y_test, (0, 3, 1, 2)))
        # pos_train = torch.tensor(y_train)
        # pos_test = torch.tensor(y_test)

        labels_path = os.path.join(data_path, f"user_positions.npy")
        labels = np.load(labels_path)

        num_samples = 252004
        IDs = np.array(range(num_samples))
        # print(len(IDs))
        trainings_size = 0.9
        # validation_size = 0.05
        test_size = 0.1

        train_IDs = IDs[:int(trainings_size*num_samples)]
        # val_IDs = IDs[int(trainings_size*num_samples):int((trainings_size + validation_size) * num_samples)]
        test_IDs = IDs[-int(test_size * num_samples):]

        # temp_x = []
        # temp_y = []
        # for i in range(num_samples):
        #     X = np.empty(dim)
        #     sample = np.load(data_path + "/samples/channel_measurement_" + str(i).zfill(6) + '.npy')
        #     X[0, :, :] = sample.real
        #     X[1, :, :] = sample.imag
        #     # X = np.transpose(X, (2, 0, 1))
        #     y = labels[i, :]

        #     X = X.astype(np.float32)
        #     y = y.astype(np.float32)

        #     temp_x.append(X)
        #     temp_y.append(y)

        
        # temp_h = [np.expand_dims(arr, axis=0) for arr in temp_x]
        # temp_pos = [np.expand_dims(arr, axis=0) for arr in temp_y]

        # H = np.concatenate(temp_h, axis=0)
        # pos = np.concatenate(temp_pos, axis=0)

        # split_index = int(trainings_size*num_samples)

        # self.train_dataset = TensorDataset(torch.tensor(H[:split_index]), torch.tensor(pos[:split_index]))
        # self.test_dataset = TensorDataset(torch.tensor(H[split_index:]), torch.tensor(pos[split_index:]))
        self.train_dataset = DataGenerator(data_path, train_IDs, labels)
        self.test_dataset = DataGenerator(data_path, test_IDs, labels, shuffle=False)


    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True,
                                  drop_last=True)
        # val_loader = DataLoader(self.val_dataset,
        #                         batch_size=self.batch_size,
        #                         num_workers=self.num_workers,
        #                         pin_memory=self.pin_memory,
        #                         shuffle=False)
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=64,
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

class DataGenerator(Dataset):
    def __init__(self, data_path, list_IDs, labels, topology="URA", batch_size=32, num_antennas=64,
                 num_subc=100, n_channels=2, shuffle=True):
        self.dim = (num_antennas, num_subc)
        if num_antennas == 64:
            self.antennas = list(range(64))
        elif num_antennas == 32:
            if topology == "DIS":
                self.antennas = [2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21,
                                 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44,
                                 45, 50, 51, 52, 53, 58, 59, 60, 61]
            elif topology == "URA":
                self.antennas = [10, 11, 12, 13, 17, 18, 19, 20, 21, 22,
                                 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38,
                                 41, 42, 43, 44, 45, 46, 50, 51, 52, 53]
            elif topology == "ULA":
                self.antennas = [x + 16 for x in range(32)]
        elif num_antennas == 16:
            if topology == "DIS":
                self.antennas = [3, 4, 11, 12, 19, 20, 27, 28,
                                 35, 36, 43, 44, 51, 52, 59, 60]
            elif topology == "URA":
                self.antennas = [18, 19, 20, 21, 26, 27, 28, 29,
                                 34, 35, 36, 37, 42, 43, 44, 45]
            elif topology == "ULA":
                self.antennas = [x + 24 for x in range(16)]
        elif num_antennas == 8:
            if topology == "DIS":
                self.antennas = [3 + 8 * x for x in range(8)]
            elif topology == "URA":
                self.antennas = [26, 27, 28, 29,
                                 34, 35, 36, 37]
            elif topology == "ULA":
                self.antennas = [x + 28 for x in range(8)]
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data_path = data_path
        # self.on_epoch_end()

    def __len__(self):
        # return int(np.floor(len(self.list_IDs) / self.batch_size))
        return len(self.list_IDs)

    def __getitem__(self, index):
        # # indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        # # list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # list_IDs_temp = self.list_IDs[index * self.batch_size: (index + 1) * self.batch_size]
        # X, y = self.__data_generation(list_IDs_temp)

        X, y = self.__data_gen(index)    

        return torch.from_numpy(X), torch.from_numpy(y)

    def __data_gen(self, index):
        X = np.empty((*self.dim, self.n_channels))
        sample = np.load(self.data_path + "/samples/channel_measurement_" + str(index).zfill(6) + '.npy')
        X[:, :, 0] = sample.real[self.antennas, :]
        X[:, :, 1] = sample.imag[self.antennas, :]
        X = np.transpose(X, (2, 0, 1))
        y = self.labels[index, :]
        
        # print(type(X), X.dtype)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        # print(type(X), X.dtype)
        # print(type(y), y.dtype)
        y = y / 1000

        return X, y


    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 3), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            sample = np.load(self.data_path + "/samples/channel_measurement_" + str(ID).zfill(6) + '.npy')
            X[i, :, :, 0] = sample.real[self.antennas, :]
            X[i, :, :, 1] = sample.imag[self.antennas, :]
            # label = np.load(self.data_path + "user_positions.npy")
            y[i] = self.labels[ID, :]

        X = np.transpose(X, (0, 3, 1, 2))

        return X, y

if __name__ == '__main__':
    root = '/mnt/HD_2/wanrongjie/KU_leuven/ultra_dense/'
    topology = "URA"
    los = 'LoS'
    datapath = root + topology + '_lab_' + los
    num_samples = 252004
    IDs = np.array(range(num_samples))
    print(len(IDs))
    trainings_size = 0.85
    train_IDs = IDs[:int(trainings_size*num_samples)]
    print(len(train_IDs))

    labels = np.load(datapath + "/user_positions.npy")
    dataset = DataGenerator(data_path=datapath, list_IDs=train_IDs, labels=labels, topology=topology)
    print(len(dataset))
    h, pos = dataset.__getitem__(0)
    print(type(h), h.shape, h.dtype)
    print(type(pos), pos.shape, pos.dtype)
    print(h)

    temp_data = np.load(datapath + "/samples/channel_measurement_000000.npy")
    print(temp_data)

    train_loader = DataLoader(dataset,
                            batch_size=1,
                            drop_last=True)
    print(next(iter(train_loader)))

    