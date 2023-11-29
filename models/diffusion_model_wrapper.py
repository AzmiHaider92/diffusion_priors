import os.path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
#from pytorch_lightning.loggers import wandb
import wandb
from data.MNIST.load_mnist import load_mnist_data
from models.diffusion import GaussianDiffusion
from models.unet import UNetModel


class diffusionModelWrapper(pl.LightningModule):
    def __init__(self,
                 args,
                 visualizefolder
                 ):
        super().__init__()

        # train parameters
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.lr = args.lr
        self.diffusion_schedule = args.diffusion_schedule
        self.diffusion_steps = args.diffusion_steps

        # build model
        self.diffusion = GaussianDiffusion(T=self.diffusion_steps, schedule=self.diffusion_schedule)
        self.unet = UNetModel(image_size=32, in_channels=1, out_channels=1,
                    model_channels=64, num_res_blocks=2, channel_mult=(1, 2, 3, 4),
                    attention_resolutions=[8, 4], num_heads=4)

        self.visualizefolder = visualizefolder
        self.visualize_every_n_epochs = args.save_ckpt_each_n_epochs

        # intermediate results
        self.epoch_loss = []

    def train_dataloader(self):
        return load_mnist_data(batch_size=self.batch_size, num_workers=self.num_workers)

    def configure_optimizers(self):
        # optimizers
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        img, labels = batch
        # Sample from the diffusion process
        t = np.random.randint(1, self.diffusion.T + 1, img.shape[0]).astype(int)
        xt, epsilon = self.diffusion.sample(img, t)
        t = torch.from_numpy(t).float().view(img.shape[0]).to(xt.device)

        # Pass through network
        out = self.unet(xt.float(), t)

        # Compute loss and backprop
        loss = F.mse_loss(out, epsilon.float())
        self.epoch_loss.append(loss.item())
        return {'loss': loss}

    def on_train_epoch_end(self):
        if self.current_epoch > 0 and self.current_epoch % self.visualize_every_n_epochs == 0:
            self.visualize_sample(self.visualizefolder)
        self.log('loss', np.mean(self.epoch_loss), on_step=False, on_epoch=True)
        self.epoch_loss = []

    def visualize_sample(self, save_path):
        diffusion = GaussianDiffusion(T=self.diffusion_steps, schedule=self.diffusion_schedule)
        # Visualize sample
        with torch.no_grad():
            self.unet.eval()
            x = diffusion.inverse(self.unet.cpu(), shape=(1, 32, 32))
            self.unet.train()

        self.logger.experiment.log(
            {"reconstruction": [wandb.Image(x.cpu().numpy()[0, 0, :, :], caption=f"epoch-{self.current_epoch}")]})

        plt.figure(figsize=(5, 5))
        plt.imshow(x.cpu().numpy()[0, 0, :, :], vmin=-1, vmax=1)
        plt.savefig(os.path.join(save_path, f'epoch-{self.current_epoch}.png'))

