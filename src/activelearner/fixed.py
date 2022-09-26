from .activelearner import ActiveLearner

import os
from scipy.io import loadmat


class FixedSampling(ActiveLearner):
    def __init__(self, dataset, args, trial):
        super().__init__(dataset, args)
        # get labeled indices
        self.indices = get_data_indices(args.data, args.data_dir, trial)


    def query(self, nQuery, model=None, idxs_lb=None):
        '''
        Query unlabeled data (selecting for active learning)
        Args:
            nQuery: number of data for query
            model: model for query
        Return:
             selected unlabeled data indices
        '''
        query_indices = self.indices[sum(idxs_lb): sum(idxs_lb) + nQuery]
        return query_indices



def get_data_indices(data_name='CIFAR10', data_dir='./dataset/', trial=0):
    data_dir = os.path.join(data_dir, 'ExtractedFeatures/')
    indices = loadmat(data_dir + f'{data_name}LabelIndex/TIdx{str(trial + 1).zfill(2)}.mat')
    indices = indices['TIdx'].squeeze() - 1
    return indices