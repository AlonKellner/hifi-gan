import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


class LitModel(pl.LightningModule):
    def __init__(self, model, learning_rate, adam_b1, adam_b2):
        super().__init__()
        self.learning_rate = learning_rate
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        y, _ = train_batch
        result = self.model(y.unsqueeze(0))
        return F.l1_loss(result, y)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            self.learning_rate,
            betas=(self.adam_b1, self.adam_b2),
            amsgrad=True
        )


