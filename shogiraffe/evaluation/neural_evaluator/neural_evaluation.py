# Imports
import shogi
from pytorch_lightning import LightningModule, Trainer
import torch


class NeuralEvaluator(LightningModule):
    def __init__(self, hidden_sizes=[256, 128, 64], lr=1e-4, dataset=None):
        super().__init__()
        self.lr = lr

        layers = []
        hidden_sizes = [784] + hidden_sizes + [1]
        for index in range(len(hidden_sizes) - 1):
            prev_dim = hidden_sizes[index]
            next_dim = hidden_sizes[index + 1]
            layers.append(torch.nn.Linear(prev_dim, next_dim))
            if index != len(hidden_sizes) - 2:
                layers.append(torch.nn.LeakyReLU(0.01))
        self.network = torch.nn.Sequential(*layers)
        self.dataset = dataset
        self.save_hyperparameters()

    def __call__(self, board):
        board_vector = None
        return self.forward(board_vector)

    def forward(self, boards_vector):
        """
        Args:
            boards_vector (torch.Tensor): the boards to evaluate, after board2vec, of shape (n_boards, 784)

        Returns:
            torch.Tensor : scores for the given boards, floats of shape (n_boards,)
        """
        return torch.sigmoid(self.network(boards_vector))

    def training_step(self, batch, batch_idx):
        scores = self.forward(batch[:, 1:])

        mse_loss = torch.nn.L1Loss()(scores, batch[:, 0:1])

        self.log('train_mae_loss', mse_loss, on_step=True, on_epoch=False)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        scores = self.forward(batch[:, 1:])

        mse_loss = torch.nn.L1Loss()(scores, batch[:, 0:1])

        self.log('val_mae_loss', mse_loss, on_step=False, on_epoch=True)
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_mae_loss"
        }