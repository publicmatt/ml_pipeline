from torch.utils.data import Dataset
import numpy as np
import einops
import csv
import torch
from pathlib import Path
from typing import Tuple
from ml_pipeline import config


class MnistDataset(Dataset):
    """
    The MNIST database of handwritten digits.
    Training set is 60k labeled examples, test is 10k examples.
    The b/w images normalized to 20x20, preserving aspect ratio.

    It's the defacto standard image training set to learn about classification in DL
    """

    def __init__(self, path: Path):
        """
        give a path to a dir that contains the following csv files:
        https://pjreddie.com/projects/mnist-in-csv/
        """
        assert path, "dataset path required"
        self.path = Path(path)
        assert self.path.exists(), f"could not find dataset path: {path}"
        self.features, self.labels = self._load()

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])

    def __len__(self):
        return len(self.features)

    def _load(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # opening the CSV file
        with open(self.path, mode="r") as file:
            images, labels = [], []
            csvFile = csv.reader(file)
            examples = config.training.examples
            for line, content in enumerate(csvFile):
                if line == examples:
                    break
                labels.append(int(content[0]))
                image = [int(x) for x in content[1:]]
                images.append(image)
            labels = torch.tensor(labels, dtype=torch.int64)
            images = torch.tensor(images, dtype=torch.float32)
            images = einops.rearrange(images, "n (w h) -> n w h", w=28, h=28)
            images = einops.repeat(
                images, "n w h -> n c (w r_w) (h r_h)", c=1, r_w=8, r_h=8
            )
            return (images, labels)


def main():
    path = Path("storage/mnist_train.csv")
    dataset = MnistDataset(path=path)
    print(f"len: {len(dataset)}")
    print(f"first shape: {dataset[0][0].shape}")
    mean = einops.reduce(dataset[:10][0], "n w h -> w h", "mean")
    print(f"mean shape: {mean.shape}")
    print(f"mean image: {mean}")


if __name__ == "__main__":
    main()
