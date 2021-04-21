import torch

from nts.data.urmp_classification import URMPClassificationDataset


def test_inception_datamodule():
    dm = URMPClassificationDataset(
        "/import/c4dm-datasets/URMP/synth-dataset/4s-dataset", ["vn", "tpt", "fl"]
    )
    assert False