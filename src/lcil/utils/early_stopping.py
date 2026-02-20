import torch.nn as nn

class EarlyStopping:
    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0
    ):
        """
        Early stopping utility to monitor a validation loss metric during 
        training and stop if it does not improve.

        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait for an improvement in the monitored metric before stopping. Default is 5.
        delta : float, optional
            Minimum change in the monitored metric to qualify as an improvement. Default is 0.0
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model: nn.Module):
        model.load_state_dict(self.best_model_state)