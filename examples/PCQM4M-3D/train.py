import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataset import PCQM4M_Dataset
from model import LitModel
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint


def main(hparams):
  dataset = PCQM4M_Dataset(grid_size=32, variance=0.4)
  train_set_size = int(.9 * len(dataset))
  validation_set_size = int(.05 * len(dataset))
  test_set_size = len(dataset) - train_set_size - validation_set_size

  train_set, validation_set, test_set = random_split(dataset=dataset,
                                                     lengths=(train_set_size,
                                                              validation_set_size,
                                                              test_set_size),
                                                      generator=torch.Generator().manual_seed(42))

  train_loader = DataLoader(train_set, batch_size = 256, pin_memory=True, num_workers=4)
  validation_loader = DataLoader(validation_set, batch_size = 256, pin_memory=True, num_workers=4)
  test_loader = DataLoader(test_set, batch_size = 256,  pin_memory=True, num_workers=4)
  model = LitModel(num_layers=10)
  wandb_logger = WandbLogger(project="ogb", entity="generativemolecules")
  wandb_logger.watch(model)
  chk_pt_dir = "/data/gpfs/Users/l1038849/ParticleGrid/examples/PCQM4M-3D/chkpts"
  checkpoint_callback = ModelCheckpoint(dirpath=chk_pt_dir, save_top_k=2, monitor="val_loss")
  trainer = pl.Trainer(logger=wandb_logger,
                       accelerator='gpu',
                       devices="auto",
                       max_epochs=400,
                       enable_model_summary=True,
                       log_every_n_steps=5,
                       strategy=DDPStrategy(find_unused_parameters=False),
                       gradient_clip_val=2,
                       track_grad_norm=2,
                       val_check_interval=.5,
                       limit_val_batches=50,
                       enable_checkpointing=True,
                       callbacks=[checkpoint_callback],
                       gradient_clip_algorithm="norm")
  # trainer.fit(model=model,
  #             train_dataloaders=train_loader,
  #             val_dataloaders=validation_loader)

  trainer.test(dataloaders=test_loader, ckpt_path='chkpts/epoch=383-step=1138995.ckpt', model=model)

if __name__ == '__main__':
  parser = ArgumentParser()
  parser = pl.Trainer.add_argparse_args(parser)
  args = parser.parse_args()
  main(args)
