from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataset import PCQM4M_Dataset
from model import LitModel
from pytorch_lightning.strategies.ddp import DDPStrategy


def main(hparams):
  dataset = PCQM4M_Dataset(grid_size=64, variance=0.4)
  train_set_size = int(.9 * len(dataset))
  validation_set_size = int(.05 * len(dataset))
  test_set_size = len(dataset) - train_set_size - validation_set_size

  train_set, validation_set, test_set = random_split(dataset=dataset,
                                                     lengths=(train_set_size,
                                                              validation_set_size,
                                                              test_set_size))

  train_loader = DataLoader(train_set, batch_size = 128, pin_memory=True, num_workers=4, prefetch_factor=4)
  validation_loader = DataLoader(validation_set, batch_size = 256, pin_memory=True, num_workers=4, prefetch_factor=4)
  test_loader = DataLoader(test_set, batch_size = 256,  pin_memory=True, num_workers=4, prefetch_factor=4)
  model = LitModel()
  wandb_logger = WandbLogger(project="ogb", entity="generativemolecules")
  wandb_logger.watch(model)
  trainer = pl.Trainer(logger=wandb_logger,
                       accelerator='gpu',
                       devices="auto",
                       max_epochs=400,
                       val_check_interval=1000,
                       enable_model_summary=True,
                       strategy=DDPStrategy(find_unused_parameters=False),)
  
  trainer.fit(model=model,
              train_dataloaders=train_loader,
              val_dataloaders=validation_loader)

  trainer.test(dataloaders=test_loader)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser = pl.Trainer.add_argparse_args(parser)
  args = parser.parse_args()
  main(args)
