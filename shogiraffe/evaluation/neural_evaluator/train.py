import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from shogiraffe.evaluation.neural_evaluator.data.data import ShogiDataModule
from shogiraffe.evaluation.neural_evaluator.neural_evaluation import NeuralEvaluator

if __name__ == "__main__":
    data_module = ShogiDataModule(
        dataset_path=os.path.join(os.path.dirname(__file__), "data/dataset/database_with_scores.txt"),
        batch_size=64,
    )
    model = NeuralEvaluator()

    wandb_logger = WandbLogger(project="shogiraffe")
    wandb_logger.watch(model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_mae_loss")

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(model, data_module)
