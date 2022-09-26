import argparse

ALL_METHODS = [
    'fixed', 'random', 'coreset', 'badge', 'ws', 'seqgcn', 'learningloss', 'vaal', 'tavaal', 'bait', 'alfamix', 'gradnorm'
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')

    parser.add_argument('-AL','--al_model', help='Active learning model', type=str, default='Ours',
                        choices=ALL_METHODS)

    parser.add_argument('-D', '--data', help='dataset', type=str, default='CIFAR10')
    parser.add_argument('--data_dir', help='dataset path', type=str, default='./dataset/')
    
    # parser.add_argument('--use_features', action='store_true', default=False, help='use pretrained feature extracted from ImageNet')
    parser.add_argument('--init', help='model initialization', type=str, default='kaiming')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix seed for reproducible')
    parser.add_argument('--seed', type=int, default=0, help='seed number')

    parser.add_argument('--start_trial', help='start trial', type=int, default=0)
    parser.add_argument('--num_trial', help='number of trials', type=int, default=10)
    parser.add_argument('--num_round', type=int, default=17, help='number of acquisition(round)')
    parser.add_argument('--num_epoch', help='number of epochs', type=int, default=100)

    # experimental settings
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size used for training only')
    parser.add_argument('--init_size', help='number of points to start', type=int, default=600)
    parser.add_argument('--query_size', help='number of points to query in a batch', type=int, default=1000)
    parser.add_argument('--labeled_data_size_per_round', help='number of points at each round (for debugging)',
                        type=list, default=[600, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                                            11000, 12000, 13000, 14000, 15000])
    
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wdecay', type=float, default=5e-4)
    parser.add_argument('--lrscheduler', type=str, default='multistep')
    parser.add_argument('--milestone', type=str, default='160')

    # additional
    parser.add_argument('--resize', help='resize image data', type=int, default=32)
    parser.add_argument('--transform', action='store_true', default=True, help='data transform')

    # save
    parser.add_argument('--no_wandb', action='store_true', default=False, help='do not use wandb')

    ## AL models
    # WS
    parser.add_argument('--ws_sampling_type', type=str, default='descending',
                        choices=['descending', 'ascending', 'sampled_descending'])
    # SeqGCN
    parser.add_argument('--lambda_loss', type=float, default=1.2,
                        help="Adjustment graph loss parameter between the labeled and unlabeled")
    parser.add_argument('--s_margin', type=float, default=0.1, help="Confidence margin of graph")
    parser.add_argument('--hidden_units', type=int, default=128, help="Number of hidden units of the graph")
    parser.add_argument('--dropout_rate', type=float, default=0.3, help="Dropout rate of the graph neural network")
    parser.add_argument('--lr_gcn', type=float, default=0.001, help="learning rate for GCN")

    # learning loss
    # for learning loss
    parser.add_argument('--epoch_loss', type=int, default=120,
                        help='After 120 epochs, stop the gradient from the loss prediction module propagated \
                                to the target model (for learning loss and tavaal)')
    parser.add_argument('--margin', type=float, default=1.0, help='MARGIN')

    # for learning loss, tavaal, and seqgcn
    parser.add_argument('--weight', '-w', type=float, default=1.0, help='weight for module loss (for learning loss, tavaal)')

    parser.add_argument('--subset_size', type=int, default=None, help='subset size of unlabeled data')

    # vaal and tavaal
    parser.add_argument('--num_epoch_vaal', help='number of epochs of vaal', type=int, default=100)
    parser.add_argument('--batch_size_vaal', help='batch size of vaal', type=int, default=64)
    
    # AlphaMix
    parser.add_argument('--alpha_cap', type=float, default=0.03125)
    parser.add_argument('--alpha_opt', action="store_const", default=False, const=True)
    parser.add_argument('--alpha_closed_form_approx', action="store_const", default=False, const=True)

    args = parser.parse_args()

    return args


'''#edict for ipynb
from easydict import EasyDict as edict

args = edict({
    'gpu_id': '0',
    'no_wandb': False,
    'al_model': 'badge',
    # dataset
    'data': 'FashionMNIST',
    'data_dir': './dataset/',
    'use_features': False,
    'init': 'kaiming',
    'resize': 32,
    'transform': False,
    # training settings
    'start_trial': 0,
    'num_trial': 10,
    'num_round': 17,
    'num_epoch': 200,
    'batch_size': 128,
    'init_size': 600,
    'query_size': 1000,
    'labeled_data_size_per_round': [600, 800, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000],
    'fix_seed': False,
    'seed': 0,
    # optimization
    'optimizer': 'sgd',
    'lr': 0.1,
    'momentum': 0.9,
    'wdecay': 5e-4,
    'lrscheduler': 'multistep',
    'milestone': None,
    # WS
    'ws_sampling_type': 'descending',
    # SeqGCN
    'lambda_loss': 1.2,
    's_margin': 0.1,
    'hidden_units': 128,
    'dropout_rate': 0.3,
    'lr_gcn': 0.001,
    # learning loss
    'epoch_loss': 120,
    'margin': 1.0,
    # learning loss, tavaal, and seqgcn
    'weight': 1.0,
    'subset_size': None,
    # vaal and tavaal
    'num_epoch_vaal': 100,
    'batch_size_vaal': 128
})
'''