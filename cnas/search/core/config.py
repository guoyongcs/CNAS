import argparse


parser = argparse.ArgumentParser("Curriculum Neural Architecture Search.")
parser.add_argument('--mode', type=str, default='CNAS_OP',
                    help='variants of CNAS ["CNAS_NODE", "CNAS_OP", "CNAS_FIX"]')
parser.add_argument('--search_space', type=str, default='DARTS_SPACE',
                    help='type of search space')
# # dataset
parser.add_argument('--data', type=str, default='.data',
                    help='location of the data corpus')
parser.add_argument('--train_portion', type=float, default=0.4,
                    help='train set proportion')

parser.add_argument('--batch_size', type=int, default=96,
                    help='batch size')
parser.add_argument('--epochs_for_warmup', type=int, default=20,
                    help='epochs_for_warmup')
parser.add_argument('--epoch_per_stage', type=int, default=40,
                    help='epoch_per_stage')
parser.add_argument('--master_start_traning_epoch', type=int, default=0,
                    help='master_start_traning_epoch')
parser.add_argument('--entropy_coeff', type=float, default=0.005,
                    help='entropy_coeff')
parser.add_argument('--grad_clip', type=int, default=5,
                    help='grad_clip')

parser.add_argument('--update_w_force_uniform', action='store_true', default=False,
                    help='update_w_force_uniform')
# # optimizer

parser.add_argument('--arch_learning_rate', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--arch_momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--arch_weight_decay', type=float, default=3e-4,
                    help='weight decay')

parser.add_argument('--controller_learning_rate', type=float, default=3e-4,
                    help='learning rate')

parser.add_argument('--controller_weight_decay', type=float, default=5e-4,
                    help='weight decay')

# architecture
parser.add_argument('--n_nodes', type=int, default=4,
                    help='num of nodes in cell architectures')
parser.add_argument('--n_ops', type=int, default=8,
                    help='num of operations in cell architectures')
parser.add_argument('--init_channels', type=int, default=16,
                    help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--multiplier', type=float, default=4,
                    help='multiplier in cell architectures')
parser.add_argument('--stem_multiplier', type=float, default=3,
                    help='stem multiplier in cell architectures')
parser.add_argument('--loose_end', action='store_true', default=False,
                    help='loose_end')
# controller
parser.add_argument('--hidden_size', type=int, default=128,
                    help='hidden size of neurons in the controller')
parser.add_argument('--temperature', type=float, default=None,
                    help='temperature')
parser.add_argument('--tanh_constant', type=float, default=None,
                    help='tanh_constant')
parser.add_argument('--op_tanh_reduce', type=float, default=None,
                    help='op_tanh_reduce')
# others
parser.add_argument('--seed', type=int, default=2020,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=2,
                    help='num_workers')
parser.add_argument('--baseline_moving_gamma', type=float, default=0.99,
                    help='baseline_moving_gamma')
parser.add_argument('--report_freq', type=float, default=50,
                    help='report frequency')

parser.add_argument('--debug', action='store_true', default=False,
                    help='debug')

args, _ = parser.parse_known_args()
