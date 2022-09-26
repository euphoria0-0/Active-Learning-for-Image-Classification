class ActiveLearner(object):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.device = args.device
        self.kwargs = {'batch_size': args.batch_size, 'pin_memory': True, 'shuffle': False}


    def query(self, nQuery, model):
        '''
        Query unlabeled data (selecting for active learning)
        Args:
            nQuery: number of data for query
            model: model for query
        Return:
             selected unlabeled data indices
        '''
        pass