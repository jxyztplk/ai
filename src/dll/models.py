import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from dll import utils
from dll.utils import HyperParameters, ProgressBoard


class Module(nn.Module, HyperParameters):
    """The base class of all models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplemented

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural nerwork is defined.'
        return self.net(X)
    
    def plot(self, key, value, train):
        """Plot a point in animation"""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, utils.numpy(utils.to(value, utils.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)


class Classifier(Module):
    """The base class of classification models."""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the number of correct predictions."""
        Y_hat = utils.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = utils.astype(utils.argmax(Y_hat, axis=1), Y.dtype)
        compare = utils.astype(preds == utils.reshape(Y, -1), utils.float32)
        return utils.reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = utils.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = utils.reshape(Y, (-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        X = utils.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

