import os

import gin
import numpy as np
import pytorch_lightning as pl
import torch

from .general import GeneralDataModule


@gin.configurable
class URMPDataModule(GeneralDataModule):
    def __init__(
        self,
        urmp_root: str,
        instrument: str,
        batch_size: int = 16,
        load_to_memory: bool = True,
        **dataloader_args
    ):
        super().__init__(
            os.path.join(urmp_root, instrument),
            batch_size,
            load_to_memory,
            **dataloader_args
        )
