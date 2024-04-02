import os
import numpy as np
# import scipy.io as sio
from pathlib import Path
import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

from utils.data import *

__all__ = ['DeepMIMODataLoader', 'PreFetcher']


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
    

class DeepMIMODataLoader(object):
    r""" PyTorch DataLoader for DeepMIMO dataset.
    """

    def __init__(self, root, batch_size, num_workers, pin_memory, scenario):
        assert os.path.isdir(root)
        root = Path(root)
        assert scenario in {"random", "wide", "narrow", "whthin"}
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        channel, nt, nc = 2, 64, 100
        
        data_root = os.path.join(root, f"DeepMIMO/I1_2p5_dataset/channel_sample/")
        number_sample = 80601
        train_size = 0.9
        test_size = 0.1
        
        labels_root = os.path.join(root, f'DeepMIMO/I1_2p5_dataset/user_location.npy')
        labels = np.load(labels_root)
        np.random.seed(2023)
        shuffle_index = np.random.permutation(number_sample)

        train_ID = shuffle_index[:int(number_sample * train_size)]
        test_ID = shuffle_index[-int(number_sample * test_size):]

        self.train_dataset = DeepMIMODataset(data_root, train_ID, labels)
        self.test_dataset = DeepMIMODataset(data_root, test_ID, labels)

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

class DeepMIMODataset(Dataset):
    def __init__(self, data_path, IDs, labels):
        super(DeepMIMODataset, self).__init__()
        self.data_path = data_path
        self.IDs = IDs
        self.labels = labels

    def __len__(self):
        return len(self.IDs)
    
    def __getitem__(self, index):
        
        temp_h = np.load(self.data_path + "sample_" + str(index).zfill(5) + ".npy")
        assert (temp_h.shape == (1, 64, 512))
        temp_h = Gaussian_noise(temp_h, snr_dB=0)

        H = np.empty((2, 64, 512))
        H[0, :, :] = temp_h.real
        H[1, :, :] = temp_h.imag

        pos = self.labels[index, :]
        pos = np.delete(pos, 2)
        pos = pos / 10
        H = H.astype(np.float32)
        pos = pos.astype(np.float32)

        return torch.from_numpy(H), torch.from_numpy(pos)
    
def Gaussian_noise(h_data, snr_dB = 20):
    # phases = np.angle(h_data)
    # amplitudes = np.abs(h_data)
    # snr_linear = 10 ** (snr_dB / 10)
    # # 计算信号功率
    # signal_power = np.mean(amplitudes ** 2)
    # noise_power = signal_power / snr_linear

    # # 定义幅值的标准差
    # amplitude_stddev = np.sqrt(noise_power)

    # # 对幅值施加高斯噪声
    # noisy_amplitudes = amplitudes * np.random.normal(1, amplitude_stddev, len(amplitudes))

    # mean = 0  # 高斯分布的均值
    # stddev = np.sqrt(noise_power)  # 高斯分布的标准差
    # noisy_phases = phases + np.random.normal(mean, stddev, len(phases))

    # noisy_complex_numbers = noisy_amplitudes * (np.cos(noisy_phases) + 1j * np.sin(noisy_phases))
    # return noisy_complex_numbers
    shape = h_data.shape
    amplitudes = np.abs(h_data)
    signal_power = np.mean(amplitudes ** 2)

    N0 = 10 ** (-snr_dB / 10)
    noise_power = np.sqrt((signal_power / 2 )* N0)

    noise = noise_power * (np.random.normal(0, 1, shape) + 1j * np.random.normal(0, 1, shape))
    noisy_data = h_data + noise
    return noisy_data

    




