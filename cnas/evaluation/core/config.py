import argparse


parser = argparse.ArgumentParser("Evaluation of NAS searched model.")
# dataset
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='[cifar10 | imagenet]')
parser.add_argument('--data', type=str, default='.data',
                    help='location of the data corpus')
# data preprocessing
parser.add_argument('--cutout_length', type=int, default=16,
                    help='cutout length (only available for cifar10)')
parser.add_argument('--eval_size', type=int, default=224,
                    help='the size of input images (only available for imagenet)')
# common setting
parser.add_argument('--max_epochs', type=int, default=600,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=96,
                    help='batch size')

parser.add_argument('--label_smooth', action='store_true', default=False,
                    help='label_smooth')
# optimizer
parser.add_argument('--learning_rate', type=float, default=0.025,
                    help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4,
                    help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5,
                    help='gradient clipping')
parser.add_argument('--scheduler', type=str, default='naive_cosine',
                    help='type of LR scheduler')
parser.add_argument('--learning_rate_min', type=float, default=0.001,
                    help='min learning rate')
parser.add_argument('--warmup_epochs', type=int, default=5,
                    help='number of epochs for warmup')
parser.add_argument('--no_bias_decay', action='store_true', default=False,
                    help='no bias decay')
# model
parser.add_argument('--arch', type=str, default='DARTS',
                    help='which architecture to use')
parser.add_argument('--init_channels', type=int, default=36,
                    help='num of init channels')
parser.add_argument('--layers', type=int, default=20,
                    help='total number of layers')
parser.add_argument('--auxiliary_weight', type=float, default=0.4,
                    help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.2,
                    help='drop path probability')
# others
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--resume', type=str, default=None,
                    help='the path to checkpoint')
parser.add_argument('--num_workers', type=int, default=2,
                    help='num_workers')
parser.add_argument('--report_freq', type=float, default=50,
                    help='report frequency')

args,_ = parser.parse_known_args()
