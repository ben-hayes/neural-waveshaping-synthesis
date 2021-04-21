import os
from typing import Union

import gin
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

from nts.utils import unzip


class URMPClassificationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_path: str,
        instruments: Union[str],
        split: str = "train",
        sample_rate: float = 16000,
    ):
        super().__init__()

        split_paths = [
            (os.path.join(root_path, instrument, split, "audio"), instrument)
            for instrument in instruments
        ]
        file_list = [
            (os.path.join(path, f), instrument)
            for path, instrument in split_paths
            for f in os.listdir(path)
        ]
        self.files, instruments = unzip(file_list)
        enc = OneHotEncoder(sparse=False)
        self.labels = enc.fit_transform(np.array(instruments)[:, np.newaxis])

        self.mel_spectrogram = MelSpectrogram(sample_rate, n_fft=1024, hop_length=256)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = np.load(self.files[idx])
        audio = torch.tensor(audio)
        print(audio.shape)
        label = self.labels[idx]

        spectrogram = self.mel_spectrogram(audio)

        return {
            "label": label,
            "spectrogram": spectrogram,
        }


class URMPClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        urmp_root: str,
        instruments: Union[str],
        batch_size: int = 16,
        sample_rate: float = 16000,
        **dataloader_args
    ):
        super().__init__()
        self.urmp_root = urmp_root
        self.instruments = instruments
        self.batch_size = batch_size
        self.dataloader_args = dataloader_args
    
    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train = URMPClassificationDataset(self.urmp_root, self.instruments, "train", sample_rate=self.sample_rate)
            self.val = URMPClassificationDataset(self.urmp_root, self.instruments, "val", sample_rate=self.sample_rate)
        elif stage == "test" or stage is None:
            self.test = URMPClassificationDataset(self.urmp_root, self.instruments, "test", sample_rate=self.sample_rate)

    def _make_dataloader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset, self.batch_size, **self.dataloader_args
        )

    def train_dataloader(self):
        return self._make_dataloader(self.train)

    def val_dataloader(self):
        return self._make_dataloader(self.val)

    def test_dataloader(self):
        return self._make_dataloader(self.test)


