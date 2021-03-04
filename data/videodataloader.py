from torch.utils.data import DataLoader
import pytorch_lightning as pl


class VideoDataloader(pl.LightningDataModule):
    def __init__(self, batch_size, dataset):
        super(VideoDataloader, self).__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_ssize, shuffle=False)