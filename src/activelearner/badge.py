'''
Reference:
    https://github.com/JordanAsh/badge
'''
from .activelearner import ActiveLearner

import sys
import pdb
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from scipy import stats
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class BADGE(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.nTrain = args.nTrain
        self.nClass = args.nClass

    def query(self, nQuery, model, idxs_lb=None):
        '''
        query data
        '''
        # dataloader
        unlabel_idxs = np.where(idxs_lb == False)[0]
        unlabel_set = Subset(self.dataset['unlabeled'], unlabel_idxs)
        unlabel_loader = DataLoader(unlabel_set, **self.kwargs)

        # get embedding
        model['backbone'].eval()
        embDim = model['backbone'].get_embedding_dim()
        embedding = np.zeros([self.nTrain, embDim * self.nClass])
        with torch.no_grad():
            for input, _, idxs in tqdm(unlabel_loader, desc='get embedding', ncols=80):
                input = input.to(self.device)

                cout = model['backbone'](input)
                out = model['backbone'].get_embedding()
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)

                # for j in range(len(input)):
                #     for c in range(self.nClass):
                #         if c == maxInds[j]:
                #             embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                #         else:
                #             embedding[idxs[j]][embDim * c: embDim * (c + 1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])

                for j in range(len(input)):
                    emb = deepcopy(out[j])
                    embedding[idxs[j]][:] = - np.tile(emb, self.nClass) * np.repeat(batchProbs[j], embDim)
                    
                    embedding[idxs[j]][embDim * maxInds[j] : embDim * (maxInds[j]+1)] += emb

        # init centers
        query_indices = init_centers(embedding[unlabel_idxs], nQuery)

        # update query data
        query_indices = np.take(unlabel_idxs, query_indices)
        
        return query_indices


# kmeans ++ initialization
def init_centers(X, K):
    ind = np.argmax([np.linalg.norm(s, 2) for s in X])
    mu = [X[ind]]
    indsAll = [ind]
    print('len X {}'.format(X.shape))
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances(X, mu).ravel().astype(float)
        else:
            newD = pairwise_distances(X, [mu[-1]]).ravel().astype(float)
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        sys.stdout.write('\r{}\t{:.10f}'.format(len(mu), sum(D2)))
        sys.stdout.flush()
        if len(mu) % 100 == 0:
            print('{}\t{:.10f}'.format(len(mu), sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        mu.append(X[ind])
        indsAll.append(ind)
        cent += 1
    '''gram = np.matmul(X[indsAll], X[indsAll].T)
    val, _ = np.linalg.eig(gram)
    val = np.abs(val)
    vgt = val[val > 1e-2]'''
    print()
    return indsAll



# kmeans ++ initialization
def init_centers_torch(X, K, device='cuda:0'):
    X = torch.tensor(X).to(device)
    ind = torch.argmax(torch.tensor([torch.linalg.norm(s, 2) for s in X])).item()
    mu = X[ind].unsqueeze(0)
    indsAll = [ind]
    print('len X {}'.format(X.shape))
    centInds = [0.] * len(X)
    cent = 0
    print('#Samps\tTotal Distance')
    pairwise_dist = torch.nn.PairwiseDistance(p=2)
    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_dist(X, mu).ravel()
        else:
            newD = pairwise_dist(X, mu[-1]).ravel()
            for i in range(len(X)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        sys.stdout.write('\r{}\t{:.10f}'.format(len(mu), sum(D2)))
        sys.stdout.flush()
        if len(mu) % 100 == 0:
            print('{}\t{:.10f}'.format(len(mu), sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel()
        Ddist = (D2 ** 2) / sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist.cpu().numpy()))
        ind = customDist.rvs(size=1)[0]
        mu = torch.cat([mu, X[ind].unsqueeze(0)], dim=0)
        indsAll.append(ind)
        cent += 1

    print()
    return indsAll