import torch

from nts.data.urmp_classification import URMPClassificationDataset
from nts.evaluation.inception import InceptionSingleChannel, InceptionInstrumentClassifier


def test_inception_model():
    model = InceptionSingleChannel(num_classes=3)
    x = torch.randn(8, 1, 299, 299)
    outputs = model(x)
    print(outputs)
    logits, aux_logits = outputs
    print(logits)
    assert False


def test_inception_datamodule():
    dm = URMPClassificationDataset(
        "/import/c4dm-datasets/URMP/synth-dataset/4s-dataset", ["vn", "tpt", "fl"]
    )
    dm[0]