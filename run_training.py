import argparse

from train import train_scl

parser = argparse.ArgumentParser(description='Sigmoid Contrastive Learning')
parser.add_argument('--dataset_path',
                    default='./data',
                    help='Path where datasets will be saved')
parser.add_argument('--dataset_name',
                    default='stl10',
                    help='Dataset name',
                    choices=['stl10', 'cifar10', "tiny_imagenet",
                             "food101", "imagenet1k"])
parser.add_argument(
    '-m',
    '--encoder_model_name',
    default='resnet18',
    choices=['resnet18', 'resnet50', "efficientnet"],
    help=
    'model architecture: resnet18, resnet50 or efficientnet (default: resnet18)'
)
parser.add_argument('-save_model_dir',
                    default='./models',
                    help='Path where models')
parser.add_argument('--num_epochs',
                    default=100,
                    type=int,
                    help='Number of epochs for training')
parser.add_argument('-b',
                    '--batch_size',
                    default=256,
                    type=int,
                    help='Batch size')
parser.add_argument('-mb',
                    '--mini_batch_size',
                    default=-1,
                    type=int,
                    help='Optionally split batch views into mini batches. ' \
                    'Similarily to gradient accumulation, enables (significantly) larger batch sizes.' \
                    'Could be further utilized for parallel or distributed mini batch processing, which '\
                    'would require multiple GPUs (may add at some point).')
parser.add_argument('-lr', '--learning_rate', default=3e-4, type=float)
parser.add_argument('-wd', '--weight_decay', default=1e-5, type=float)
parser.add_argument('--fp16_precision',
                    action='store_true',
                    help='Whether to use 16-bit precision for GPU training')

parser.add_argument('--proj_out_dim',
                    default=64,
                    type=int,
                    help='Projector MLP out dimension')
parser.add_argument('--proj_hidden_dim',
                    default=512,
                    type=int,
                    help='Projector MLP hidden dimension')
parser.add_argument('--log_every_n_steps',
                    default=400,
                    type=int,
                    help='Log every n steps')
parser.add_argument('--beta',
                    default=0.995,
                    type=float,
                    help='Initial EMA coefficient')
parser.add_argument('--alpha',
                    default=1.0,
                    type=float,
                    help='Regularization loss factor')
parser.add_argument('--update_beta_after_step',
                    default=1,
                    type=int,
                    help='Update EMA beta after this step')
parser.add_argument('--update_beta_every_n_steps',
                    default=1,
                    type=int,
                    help='Update EMA beta after this many steps')
parser.add_argument('--gamma',
                    default=1.0,
                    type=float,
                    help='Initial confidence penalty')
parser.add_argument('--gamma_scaling_steps',
                    default=20_000,
                    type=int,
                    help='Number of first N steps during which gamma will be scaled from the inital value to 0')
parser.add_argument('--use_gamma_scaling',
                    action='store_true',
                    help='Whether to use gamma(conf penalty) cosine scaling')
parser.add_argument('--ckpt_path',
                    default=None,
                    type=str,
                    help='Specify path to scl_model.pth to resume training')
parser.add_argument('--num_global_views',
                    default=2,
                    type=int,
                    help='Number of global (large) views to generate through augmentation')
parser.add_argument('--num_local_views',
                    default=4,
                    type=int,
                    help='Number of local (small) views to generate through augmentation')

parser.add_argument('--use_penalty_filtering',
                    action='store_true',
                    help='If True, explicitly exclude pairs that are below filtering_threshold')
parser.add_argument('--filtering_threshold',
                    default=0.05,
                    type=float,
                    help='Keep only pairs that have a confidence penalty above this threshold')
parser.add_argument('--filtering_warmup_steps',
                    default=5000,
                    type=int,
                    help='Warm up with standard confidence penalty (no explicit filtering) for this many steps.')


def main():
    args = parser.parse_args()
    train_scl(args)


if __name__ == "__main__":
    main()