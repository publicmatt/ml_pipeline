from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer


class Runner:
    """Runner class that is in charge of implementing routine training functions such as running epochs or doing inference time"""

    def __init__(self, train_set: Dataset, train_loader: DataLoader, model: nn.Module, optimizer: Optimizer):
        # Initialize class attributes
        self.train_set = train_set

        # Prepare opt, model, and train_loader (helps accelerator auto-cast to devices)
        self.optimizer, self.model, self.train_loader = (
            optimizer, model, train_loader
        )

        # Since data is for targets, use Mean Squared Error Loss
        # self.criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()

    def step(self):
        """Runs an epoch of training.

        Includes updating model weights and tracking training loss

        Returns:
            float: The loss averaged over the entire epoch
        """

        # turn the model to training mode (affects batchnorm and dropout)
        self.model.train()

        total_loss, total_samples = 0.0, 0.0
        for sample, target in self.train_loader:
            self.optimizer.zero_grad()  # reset gradients to 0
            prediction = self.model(sample)  # forward pass through model
            loss = self.criterion(prediction, target)  # error calculation

            # increment gradients within model by sending loss backwards
            loss.backward()
            self.optimizer.step()  # update model weights

            total_loss += loss # increment running loss
            total_samples += len(sample)
            yield total_loss / total_samples  # take the average of the loss over each sample
