import configargparse


def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # logging
    parser.add_argument("--log", type=str, default='./log',
                        help='log directory')
    parser.add_argument("--expname", type=str, default='tmp',
                        help='experiment name')
    parser.add_argument("--add_timestamp", type=bool, default=True,
                        help='to the log folder')

    # train parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help='batch size')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument("--diffusion_schedule", type=str, default="linear",
                        help='linear vs cosine')
    parser.add_argument("--diffusion_steps", type=int, default=1000,
                        help='T in diffusion model')


    # pytorch lightning parameters
    parser.add_argument("--num_workers", type=int, default=2,
                        help='for data loader')
    parser.add_argument("--strategy", type=str, default="dp",
                        help='pl strategy, single/multiple devices')
    parser.add_argument("--n_epochs", type=int, default=-1,
                        help='number of epochs to run')
    parser.add_argument("--log_every_n_steps", type=int, default=10,
                        help='log to wandb every n steps (batches)')
    parser.add_argument("--save_top_k", type=int, default=3,
                        help='saving top k checkpoints')
    parser.add_argument("--save_ckpt_each_n_epochs", type=int, default=20,
                        help='save a checkpoint every n epochs')


    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()