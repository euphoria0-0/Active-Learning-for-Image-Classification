'''
Reference:
	https://github.com/xulabs/aitom/tree/master/aitom/ml/active_learning/al_gradnorm
'''
from .activelearner import ActiveLearner

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm


class GradNorm(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.subset_size = args.subset_size
        self.nClass = args.nClass
        self.scheme = 'expected'  # expected OR entropy
        self.kwargs['batch_size'] = 1
         
    def query(self, nQuery, model, idxs_lb=None):
        '''
		query data
		'''
        # dataloader
        unlabeled_idxs = np.where(idxs_lb == False)[0]
        if self.subset_size is None or self.subset_size > len(unlabeled_idxs):
            subset = unlabeled_idxs
        else:
            subset = np.random.choice(unlabeled_idxs, self.subset_size, replace=False)

        unlabeled_set = Subset(self.dataset['unlabeled'], subset)
        unlabel_loader = DataLoader(unlabeled_set, **self.kwargs)

        model['backbone'].eval()
        criterion = nn.CrossEntropyLoss()
        uncertainty = torch.tensor([]).to(self.device)

        # get uncertainty based on gradient norm
        for inputs, _, _ in tqdm(unlabel_loader, desc='get gradient_norm', ncols=80):
            inputs = inputs.to(self.device)

            scores = model['backbone'](inputs)
            posterior = F.softmax(scores, dim=1)

            loss = 0.0

            if self.scheme == 'expected':   # expected-gradnorm
                posterior = posterior.squeeze()

                for i in range(self.nClass):
                    label = torch.full([1], i)
                    label = label.to(self.device)
                    loss += posterior[i] * criterion(scores, label)

            else:  # entropy-gradnorm
                loss = Categorical(probs=posterior).entropy()

            # compute gradient norm
            pred_gradnorm = self.compute_gradnorm(model['backbone'], loss)
            pred_gradnorm = torch.sum(pred_gradnorm)
            pred_gradnorm = pred_gradnorm.unsqueeze(0)

            uncertainty = torch.cat((uncertainty, pred_gradnorm), 0)

            torch.cuda.empty_cache()

        query_indices = np.take(subset, torch.argsort(uncertainty)[-nQuery:].numpy())

        return query_indices


    def compute_gradnorm(self, model, loss):
        grad_norm = torch.tensor([]).to(self.device)
        gradnorm = 0.0

        model.zero_grad()
        loss.backward(retain_graph=True)
        for param in model.parameters():
            if param.grad is not None:
                gradnorm = torch.norm(param.grad)
                gradnorm = gradnorm.unsqueeze(0)
                grad_norm = torch.cat((grad_norm, gradnorm), 0)

        return grad_norm