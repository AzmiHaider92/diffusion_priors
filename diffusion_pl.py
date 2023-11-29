import datetime
import shutil

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.diffusion_model_wrapper import diffusionModelWrapper
from opt import config_parser
from models.unet import UNetModel
from models.diffusion import GaussianDiffusion
import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
mpl.rc('image', cmap='gray')

# if cuda is available, use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


def sample_diffusion(diffusion, data_loader, batch_size):
    x0, y = next(iter(data_loader))
    t = np.random.randint(1, diffusion.T + 1, batch_size).astype(int)
    xt, _ = diffusion.sample(x0, t)

    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
    for i in range(4):
        ax[0, i].imshow(x0[i, 0, :, :].numpy(), vmin=-1, vmax=1)
        ax[0, i].set_title(str(y[i].item()))

        ax[1, i].imshow(xt[i, 0, :, :].numpy(), vmin=-1, vmax=1)
        ax[1, i].set_title(f't={t[i]}')
    plt.show()


def sample_from_learned_diffusion(net):
    # Sample from the learned diffusion process
    diffusion = GaussianDiffusion(T=1000, schedule='linear')

    with torch.no_grad():
        net.eval()
        x = diffusion.inverse(net, shape=(1, 32, 32), device=device)
        net.train()

    plt.figure(figsize=(5, 5))
    plt.imshow(x.cpu().numpy()[0, 0, :, :], vmin=-1, vmax=1)
    plt.show()


def log_hyperparameters(configFile, logger):
    print("====================> HyperParameters <====================")
    with open(file=configFile, mode='r') as fs:
        for l in fs.readlines():
            if l[0] in ["#", "\n", "/n"]:
                continue
            k, v = l.strip().split('=')
            logger.experiment.config.update({k: v})
            print(f"{k} : {v}")
    print("====================> HyperParameters <====================")


if __name__ == "__main__":

    hparams = config_parser()
    # logging
    if hparams.add_timestamp:
        run_name = f'{hparams.expname}{datetime.datetime.now().strftime("-M%m-D%d-H%H-M%M")}'
        logfolder = f'{hparams.log}/{run_name}'
    else:
        run_name = f'{hparams.expname}'
        logfolder = f'{hparams.log}/{run_name}'
    checkpointfolder = f'{logfolder}/checkpoints'
    configfolder = f'{logfolder}/config'
    visualizefolder = f'{logfolder}/vis'

    # init log file and checkpoints
    os.makedirs(logfolder, exist_ok=True)
    os.makedirs(checkpointfolder, exist_ok=True)
    os.makedirs(configfolder, exist_ok=True)
    os.makedirs(visualizefolder, exist_ok=True)
    # copy configuration file to log folder
    shutil.copyfile(hparams.config, os.path.join(configfolder, run_name + '-config.txt'))

    wandb_logger = WandbLogger(# set the wandb project where this run will be logged
        project="Diffusion",
        name=run_name,
        dir=logfolder)

    # add hyperparameters to config
    log_hyperparameters(hparams.config, wandb_logger)

    # model
    model = diffusionModelWrapper(hparams,
                               visualizefolder)

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
    else:
        accelerator = "cpu"
        devices = 0

    # saving a checkpoint
    checkpoint_callback = ModelCheckpoint(dirpath=checkpointfolder,
                                          filename='{epoch}-{loss:.2f}',
                                          monitor="loss",
                                          mode="min",
                                          save_top_k=3,
                                          auto_insert_metric_name=True,
                                          every_n_epochs=hparams.save_ckpt_each_n_epochs,
                                          save_last=True)

    trainer = Trainer(logger=wandb_logger,
                      strategy=hparams.strategy,
                      accelerator=accelerator,
                      devices=devices,
                      max_epochs=hparams.n_epochs,
                      log_every_n_steps=hparams.log_every_n_steps,
                      check_val_every_n_epoch=hparams.save_ckpt_each_n_epochs,
                      enable_checkpointing=True,
                      detect_anomaly=True)

    # fit
    trainer.fit(model)

    print("done")