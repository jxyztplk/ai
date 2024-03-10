from dll.utils import HyperParameters

DATA_PATH = '../../data'

class DataModule(HyperParameters):
    """The base class of data."""
    def __init__(self, root=DATA_PATH, num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)