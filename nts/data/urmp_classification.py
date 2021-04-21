import os
from typing import Union

import gin
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import OneHotEncoder
import torch
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
            (os.path.join(root_path, instrument, split), instrument)
            for instrument in instruments
        ]
        file_list = [
            (os.path.join(path, f), instrument)
            for path, instrument in split_paths
            for f in os.listdir(os.path.join(path, "audio"))
        ]
        self.files, instruments = unzip(file_list)
        enc = OneHotEncoder(sparse=False)
        self.labels = enc.fit_transform(np.array(instruments)[:, np.newaxis])

        self.mel_spectrogram = MelSpectrogram()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio = np.load(self.files[idx])
        label = self.labels[idx]

        # spectrogram =
