from ml_pipeline.data.dataset import MnistDataset
from ml_pipeline import config
from pathlib import Path
import pytest

@pytest.mark.skip()
def test_init():
    pass


def test_getitem():
    train_set = MnistDataset(config.data.train_path)

    assert train_set[0][1].item() == 5
    repeated = 8
    length = 28
    channels = 1
    assert train_set[0][0].shape == (channels, length * repeated, length * repeated)

@pytest.mark.skip()
def test_loader():
    from torch.utils.data import DataLoader
    train_set = MnistDataset(config.data.train_path)
    # train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True)
    # for sample, target in train_loader:
    #     assert len(sample) == config.training.batch_size
    # len(sample)
    # len(target)
