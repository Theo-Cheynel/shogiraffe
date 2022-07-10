import numpy as np
import pytorch_lightning as pl
import shogi
import torch
from tqdm import tqdm

from shogiraffe.agents.board_to_vec import board2vec


class ShogiDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size, train_val_split=(0.8, 0.1)):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.train_boards = None
        self.val_boards = None

    def setup(self, stage):
        if self.dataset_path.endswith(".txt"):
            with open(self.dataset_path, encoding="utf-8") as file:
                boards = torch.zeros((len(file.readlines()), 784 + 1))
                file.seek(0)
                for index, line in enumerate(tqdm(file.readlines())):
                    score, sfen = line.split(" | ")
                    board = shogi.Board(sfen)
                    vec = np.concatenate([np.array([float(score)]), board2vec(board)])
                    boards[index] = torch.from_numpy(vec).float()
            np.save(self.dataset_path[:-3] + "npy", boards.detach().cpu().numpy())
        else:
            boards = torch.from_numpy(np.load(self.dataset_path))

        train_amount = int(self.train_val_split[0] * len(boards))
        val_amount = int(self.train_val_split[1] * len(boards))

        # Normalizing scores
        boards[:, 0] = torch.sigmoid(boards[:, 0] / 1000)

        self.train_boards = boards[:train_amount]
        self.val_boards = boards[train_amount : train_amount + val_amount]

        scores = self.train_boards[:, 0].flatten()
        mean = torch.mean(scores)
        mse = torch.nn.L1Loss()(scores, mean.repeat(len(scores)))
        print(f"Average score : {mean}")
        print(f"Average L1 to the mean score : {mse}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_boards, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_boards, batch_size=self.batch_size, shuffle=True)
