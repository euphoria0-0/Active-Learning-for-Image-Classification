from .activelearner import ActiveLearner

import numpy as np


class RandomSampling(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)


    def query(self, nQuery, model, idxs_lb=None):
        '''
        Query unlabeled data (selecting for active learning)
        Args:
            nQuery: number of data for query
            model: model for query
        Return:
             selected unlabeled data indices
        '''
        unlabeled_indices = np.where(idxs_lb==False)[0]
        query_indices = np.random.choice(unlabeled_indices, nQuery, replace=False)
        return query_indices