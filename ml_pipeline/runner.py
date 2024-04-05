from torch import nn


class Runner:
    """Runner class that is in charge of implementing routine training functions such as running epochs or doing inference time"""

    def __init__(self, train_set, train_loader, accelerator, model, optimizer):
        # Initialize class attributes
        self.accelerator = accelerator
        self.train_set = train_set

        # Prepare opt, model, and train_loader (helps accelerator auto-cast to devices)
        self.optimizer, self.model, self.train_loader = accelerator.prepare(
            optimizer, model, train_loader
        )

        # Since data is for targets, use Mean Squared Error Loss
        self.criterion = nn.MSELoss()

    def step(self):
        """Runs an epoch of training.

        Includes updating model weights and tracking training loss

        Returns:
            float: The loss averaged over the entire epoch
        """

        # Turn the model to training mode (affects batchnorm and dropout)
        self.model.train()

        running_loss, running_sample_count = 0.0, 0.0
        for sample, target in self.train_loader:
            self.optimizer.zero_grad()  # Reset gradients to 0
            prediction = self.model(sample)  # Forward pass through model
            loss = self.criterion(prediction, target)  # Error calculation
            running_loss += loss  # Increment running loss
            running_sample_count += len(sample)
            self.accelerator.backward(
                loss
            )  # Increment gradients within model by sending loss backwards
            self.optimizer.step()  # Update model weights
            yield running_loss / running_sample_count  # Take the average of the loss over each sample
