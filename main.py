import numpy as np
import os
import time
import random
import platform

import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset import ImageDataset
from src.model import *
from src.trainer import Trainer
from src.utils import utils
from src.activelearner import *
from src.argparse import get_args



def get_activelearner(dataset, args, trial):
    if args.al_model == 'fixed':
        return FixedSampling(dataset, args, trial)
    elif args.al_model == 'random':
        return RandomSampling(dataset, args)
    elif args.al_model == 'coreset':
        return CoreSet(dataset, args)
    elif args.al_model == 'badge':
        return BADGE(dataset, args)
    elif args.al_model == 'ws':
        return WeightDecayScheduling(dataset, args)
    elif args.al_model == 'seqgcn':  # UncertainGCN
        assert args.subset_size is not None
        return SequentialGCN(dataset, args)
    elif args.al_model == 'coregcn': # CoreGCN
        return SequentialGCN(dataset, args, False)
    elif args.al_model == 'learningloss':
        assert args.subset_size is not None
        return LearningLoss(dataset, args)
    elif args.al_model == 'vaal':
        return VAAL(dataset, args)
    elif args.al_model == 'tavaal':
        assert args.subset_size is not None
        return TAVAAL(dataset, args)
    elif args.al_model == 'bait':
        return BAIT(dataset, args)
    elif args.al_model == 'alfamix':
        return AlphaMixSampling(dataset, args)
    elif args.al_model == 'gradnorm':
        assert args.subset_size is not None
        return GradNorm(dataset, args)


def create_models(args):
    models = {}

    # backbone: Resnet
    if args.resize == 32:
        models['backbone'] = ResNet18(num_classes=args.nClass, method=args.al_model, dataset=args.data).to(args.device)
    elif args.resize == 224:
        models['backbone'] = ResNet18_224(num_classes=args.nClass, method=args.al_model, finetune=False, use_pretrained=False)
    else:
        raise('Select image size from {32, 224}')
    
    # module
    flag_module = args.al_model in ['learningloss', 'tavaal']
    if flag_module:
        models['module'] = LossNet(init='kaiming').to(args.device)
    return models


def load_data(args):
    # use image dataset
    dataset = ImageDataset(args.data, args.data_dir, args.resize, args.transform, pretrained=False)
    args.nClass, args.nTrain = dataset.nClass, dataset.nTrain
    dataset = dataset.dataset

    return dataset, args


def main(args):
    # settings
    args.machine = platform.uname().node
    args.start = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    
    # fix seed
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # device setting
    if args.gpu_id == 'cpu' or not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        args.device = torch.device(f"cuda:{args.gpu_id[0]}")
        torch.cuda.set_device(args.device)
        print('Current cuda device ', torch.cuda.current_device())
    
    # output path to save results
    args.out_path = f'./results/{args.data}/{args.al_model}/{args.al_model}_{args.start}/'
    os.makedirs(args.out_path+'Idx/', exist_ok=True)

    result_file = utils.ResultFiles(args)
    writer = SummaryWriter(f'{args.out_path}/runs/')
    
    print('='*30, ' Output Path : ' + args.out_path, '='*30, sep='\n')

    # load dataset
    dataset, args = load_data(args)

    ## TRIAL
    QuerySize = args.query_size

    for trial in range(args.num_trial):
        print('====== TRIAL {} ======'.format(trial + 1))

        # get active learner
        activelearner = get_activelearner(dataset, args, args.start_trial + trial)

        # generate initial labeled pool
        idxs_lb = np.zeros(args.nTrain, dtype=bool)
        # randomly selected initial labeled pool
        # initial_indices = np.random.choice(args.nTrain, args.init_size, replace=False)
        # fixed selected initial labeled pool for reproducible results
        initial_sampler = FixedSampling(dataset, args, args.start_trial + trial)
        initial_indices = initial_sampler.query(args.init_size, None, idxs_lb)
        idxs_lb[initial_indices] = True

        # to save results
        result_file.new_trial_result(trial)

        # Active learning round
        for round in range(args.num_round):
            QuerySize = 200 if sum(idxs_lb) < 1000 else args.query_size

            print('>> Trial {} Round {}: #Labeled {} #Query {}'.format(trial+1, round+1, sum(idxs_lb), QuerySize))
            assert sum(idxs_lb) == args.labeled_data_size_per_round[round]  # for debugging

            # load model
            models = create_models(args)

            # training
            trainer = Trainer(dataset, models, args, writer, round)
            trainer.train(idxs_lb)

            # round test accuracy
            test_acc = trainer.test()

            # save results
            result_file.save_result(idxs_lb, test_acc, round)

            # query data
            if round < args.num_round - 1:
                query_indices = activelearner.query(QuerySize, trainer.get_model(), idxs_lb)
                idxs_lb[query_indices] = True

                # save queried index
                result_file.save_query_index(query_indices)

        # save results
        result_file.end_trial_result()
    
    # save results
    result_file.wandb_final_result()


if __name__ == '__main__':
    args = get_args()
    main(args)