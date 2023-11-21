import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, Accuracy
from config import conf
import shutil
import os


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=conf['bs'], num_workers=conf['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=conf['bs'], num_workers=conf['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=conf['bs'], num_workers=conf['num_workers'])

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=conf['bs'], num_workers=conf['num_workers'])

class LitMLP_Classifier(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, conf['hidden_size'])
        self.fc2 = nn.Linear(conf['hidden_size'], conf['hidden_size'])
        self.fc3 = nn.Linear(conf['hidden_size'], 10)
        self.dropout = nn.Dropout(conf['drop_out_p'])
        #self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.BCELoss()
        # self.loss = nn.NLLLoss()
        self.loss = nn.CrossEntropyLoss()
        self.sig = nn.Sigmoid()
        self.softmax = nn.LogSoftmax(dim=1)
        metrics = MetricCollection([Accuracy(task="multiclass", num_classes=10)])#MulticlassAccuracy(10), MulticlassPrecision(10), MulticlassRecall(10)])
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        # x = self.sig(self.fc3(x))
        x = self.softmax(self.fc3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y_sing = batch
        y = F.one_hot(y_sing, num_classes = 10).float()
        logits = self(x)
        loss = self.loss(logits, y_sing)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.train_metrics.update(logits, y_sing)
        return {"loss": loss}
    
    def on_training_epoch_end(self,outputs):
        output = self.train_metrics.compute()
        self.log_dict(output)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y_sing = batch
        y = F.one_hot(y_sing, num_classes = 10).float()
        logits = self(x)
        loss = self.loss(logits, y_sing)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.val_metrics.update(logits, y_sing)

    def on_validation_epoch_end(self):
        output = self.val_metrics.compute()
        self.log_dict(output)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y_sing = batch
        y = F.one_hot(y_sing, num_classes = 10).float()
        logits = self(x)
        loss = self.loss(logits, y_sing)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(logits, y_sing)

    def on_test_epoch_end(self):
        output = self.test_metrics.compute()
        self.log_dict(output)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=conf['lr'])
        return optimizer

def main():
    data_module = MNISTDataModule(os.getcwd())
    classifier = LitMLP_Classifier()
    logger = TensorBoardLogger(conf['save_path'], name=conf['logger']['version'])
    checkpoint_callback = ModelCheckpoint(
        monitor="val_MulticlassAccuracy",
        mode="max",
        dirpath=conf['save_path'],
        filename="mnist-{epoch:02d}-{val_MulticlassAccuracy:.2f}",
    )
    early_stop_cb = EarlyStopping(monitor='val_MulticlassAccuracy', patience=conf['patience'], mode='max')
    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])
    shutil.copy('config.py', conf['save_path']+'config.py')
    trainer = pl.Trainer(
        devices=conf['trainer']['devices'],
        max_epochs=conf['trainer']['max_epochs'],
        accelerator=conf['trainer']['accelerator'],
        strategy=conf['trainer']['strategy'],
        default_root_dir=conf['trainer']['default_root_dir'],
        logger=logger,
        callbacks=[TQDMProgressBar(refresh_rate=conf['trainer']['refresh_rate']), checkpoint_callback, early_stop_cb]
    )
    trainer.fit(classifier, data_module)

    trainer.test(ckpt_path='best', datamodule=data_module)

if __name__ == "__main__":
    main()              
