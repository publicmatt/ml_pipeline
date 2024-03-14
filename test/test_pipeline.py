from src.model.linear import DNN
from src.data.dataset import MnistDataset
import os


def test_size_of_dataset():
    examples = 500
    os.environ["TRAINING_EXAMPLES"] = str(examples)
    channels = 1
    width, height = 224, 224
    dataset = MnistDataset(os.getenv("TRAIN_PATH"))
    # label = dataset[0][1].item()
    image = dataset[0][0].shape
    assert channels == image[0]
    assert width == image[1]
    assert height == image[2]
    assert len(dataset) == examples
