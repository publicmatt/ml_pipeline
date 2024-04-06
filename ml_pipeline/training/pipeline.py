
from torch.utils.data import DataLoader
from torch.optim import AdamW
from ml_pipeline.training.runner import Runner
from ml_pipeline import config


def run():
    # Initialize the training set and a dataloader to iterate over the dataset
    # train_set = GenericDataset()
    train_set = get_dataset()
    train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=True)

    model = get_model(name=config.model.name)

    # Get the size of the input and output vectors from the training set
    # in_features, out_features = train_set.get_in_out_size()


    optimizer = AdamW(model.parameters(), lr=config.training.learning_rate)

    # Create a runner that will handle
    runner = Runner(
        train_set=train_set,
        train_loader=train_loader,
        model=model,
        optimizer=optimizer,
    )

    # Train the model
    for _ in range(config.training.epochs):
        # Run one loop of training and record the average loss
        for step in runner.step():
            print(f"{step}")

def get_model(name='vgg11'):
    from ml_pipeline.model.linear import DNN
    from ml_pipeline.model.cnn import VGG11
    if name == 'vgg11':
        return VGG11(config.data.in_channels, config.data.num_classes)
    else:
        # Create the model and optimizer and cast model to the appropriate GPU
        in_features, out_features = dataset.in_out_features()
        model = DNN(in_features, config.model.hidden_size, out_features)
    return model.to(config.training.device)


def get_dataset(source='mnist', split='train'):
    # Usage
    from ml_pipeline.data.dataset import MnistDataset
    from torchvision import transforms
    csv_file_path = config.data.train_path
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts a PIL Image or numpy.ndarray to a FloatTensor and scales the image's pixel intensity values to the [0., 1.] range
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize using the mean and std specific to MNIST
    ])

    dataset = MnistDataset(csv_file_path)
    return dataset
