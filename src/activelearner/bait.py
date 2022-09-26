'''
Reference:
    https://github.com/JordanAsh/badge
'''
from .activelearner import ActiveLearner

import sys
import gc
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader



class BAIT(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.nTrain = args.nTrain
        self.nClass = args.nClass
        self.lamb = 0.01  # 1
        self.batch_size = 1000  # should be as large as gpu memory allows
        self.probs = []
        self.over_sample = 2
        self.tqdm_opts = {'ncols': 80}

    def query(self, nQuery, model, idxs_lb=None):
        '''
        query data
        '''
        print('>> Query')
        # unlabeled data
        unlabel_idxs = np.where(idxs_lb == False)[0]
        train_set = self.dataset['unlabeled']
        self.kwargs['batch_size'] = self.batch_size
        all_train_loader = DataLoader(train_set, **self.kwargs)

        # get low-rank point-wise fishers
        # xt = self.get_exp_grad_embedding(all_train_loader)
        # fisher embedding for bait (assumes cross-entropy loss)
        model['backbone'].eval()
        embDim = model['backbone'].get_embedding_dim()
        # embedding = np.zeros([self.nTrain, self.nClass, embDim * self.nClass])
        xt = torch.zeros(self.nTrain, self.nClass, embDim * self.nClass)

        # for ind in tqdm(range(self.nClass), desc='getting embedding', **self.tqdm_opts):
        #     # unlabeled_loader = DataLoader(unlabeled_set, self.kwargs)
        #     with torch.no_grad():
        #         for input, _, idxs in all_train_loader:
        #             input = input.to(self.device)

        #             cout = model['backbone'](input)
        #             out = model['backbone'].get_embedding()
        #             # out = out.data.cpu().numpy()
        #             # batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                    
        #             # for j in range(len(input)):
        #             #     for c in range(self.nClass):
        #             #         if c == ind:
        #             #             embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
        #             #         else:
        #             #             embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
                        
        #             #     if len(self.probs) > 0: 
        #             #         embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(self.probs[idxs[j]][ind])
        #             #     else: 
        #             #         embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(batchProbs[j][ind])
                    
        #             out = out.data.cpu()
        #             batchProbs = F.softmax(cout, dim=1).data.cpu()

        #             for j in range(len(input)):
        #                 emb = deepcopy(out[j])
        #                 xt[idxs[j]][ind][:] = - emb.tile((self.nClass)) * batchProbs[j].repeat_interleave(embDim)

        #                 xt[idxs[j]][ind][embDim * ind : embDim * (ind+1)] += emb

        #                 if len(self.probs) > 0: 
        #                     xt[idxs[j]][ind] *= torch.sqrt(self.probs[idxs[j]][ind])
        #                 else: 
        #                     xt[idxs[j]][ind] *= torch.sqrt(batchProbs[j][ind])

        
        with torch.no_grad():
            for input, _, idxs in tqdm(all_train_loader, desc='getting embedding', ncols=80):
                input = input.to(self.device)

                cout = model['backbone'](input)
                out = model['backbone'].get_embedding()
                out = out.data.cpu()
                batchProbs = F.softmax(cout, dim=1).data.cpu()

                for j in tqdm(range(len(input)), desc='computing embedding', ncols=80, leave=False):
                    emb = deepcopy(out[j])

                    for ind in range(self.nClass):
                        xt[idxs[j]][ind][:] = - emb.tile((self.nClass)) * batchProbs[j].repeat_interleave(embDim)

                        xt[idxs[j]][ind][embDim * ind : embDim * (ind+1)] += emb

                        if len(self.probs) > 0: 
                            xt[idxs[j]][ind] *= torch.sqrt(self.probs[idxs[j]][ind])
                        else: 
                            xt[idxs[j]][ind] *= torch.sqrt(batchProbs[j][ind])

        # xt = torch.Tensor(embedding)

        # get fisher
        # print('getting fisher matrix...', flush=True)
        fisher = torch.zeros(xt.shape[-1], xt.shape[-1])
        rounds = int(np.ceil(self.nTrain / self.batch_size))
        for i in tqdm(range(rounds), desc='getting fisher matrix', **self.tqdm_opts):
            xt_ = xt[i * self.batch_size : (i + 1) * self.batch_size].to(self.device)
            op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt)), 0).detach().cpu()
            fisher = fisher + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        # get fisher only for samples that have been seen before
        init = torch.zeros(xt.shape[-1], xt.shape[-1])
        xt2 = xt[idxs_lb]
        rounds = int(np.ceil(len(xt2) / self.batch_size))
        for i in tqdm(range(rounds), desc='getting fisher only for labeled set', **self.tqdm_opts):
            xt_ = xt2[i * self.batch_size : (i + 1) * self.batch_size].to(self.device)
            op = torch.sum(torch.matmul(xt_.transpose(1,2), xt_) / (len(xt2)), 0).detach().cpu()
            init = init + op
            xt_ = xt_.cpu()
            del xt_, op
            torch.cuda.empty_cache()
            gc.collect()

        # select
        X = xt[unlabel_idxs]
        nLabeled = np.sum(idxs_lb)
        # numEmbs = len(X)
        indsAll = []
        dim = X.shape[-1]
        rank = X.shape[-2]

        currentInv = torch.inverse(self.lamb * torch.eye(dim).to(self.device) + init.to(self.device) * nLabeled / (nLabeled + nQuery))
        X = X * np.sqrt(nQuery / (nLabeled + nQuery))
        fisher = fisher.to(self.device)

        # forward selection, over-sample by 2x
        # print('forward selection...', flush=True)
        # over_sample = 2
        rounds = int(self.over_sample *  nQuery)
        for i in tqdm(range(rounds), desc='forward selection', **self.tqdm_opts):

            # check trace with low-rank updates (woodbury identity)
            xt_ = X.to(self.device)
            innerInv = torch.inverse(torch.eye(rank).to(self.device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
            innerInv[torch.where(torch.isinf(innerInv))] = torch.sign(innerInv[torch.where(torch.isinf(innerInv))]) * np.finfo('float32').max
            traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)

            # clear out gpu memory
            xt = xt_.cpu()
            del xt, innerInv
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

            # get the smallest unselected item
            traceEst = traceEst.detach().cpu().numpy()
            for j in np.argsort(traceEst)[::-1]:
                if j not in indsAll:
                    ind = j
                    break

            indsAll.append(ind)
            print(i, ind, traceEst[ind], flush=True)
        
            # commit to a low-rank update
            xt_ = X[ind].unsqueeze(0).to(self.device)
            innerInv = torch.inverse(torch.eye(rank).to(self.device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
            currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

        # backward pruning
        # print('backward pruning...', flush=True)
        for i in tqdm(range(len(indsAll) - nQuery), desc='backward pruning', **self.tqdm_opts):

            # select index for removal
            xt_ = X[indsAll].to(self.device)
            innerInv = torch.inverse(-1 * torch.eye(rank).to(self.device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
            traceEst = torch.diagonal(xt_ @ currentInv @ fisher @ currentInv @ xt_.transpose(1, 2) @ innerInv, dim1=-2, dim2=-1).sum(-1)
            delInd = torch.argmin(-1 * traceEst).item()
            print(len(indsAll) - i, indsAll[delInd], -1 * traceEst[delInd].item(), flush=True)

            # low-rank update (woodbury identity)
            xt_ = X[indsAll[delInd]].unsqueeze(0).to(self.device)
            innerInv = torch.inverse(-1 * torch.eye(rank).to(self.device) + xt_ @ currentInv @ xt_.transpose(1, 2)).detach()
            currentInv = (currentInv - currentInv @ xt_.transpose(1, 2) @ innerInv @ xt_ @ currentInv).detach()[0]

            del indsAll[delInd]

        del xt_, innerInv, currentInv
        torch.cuda.empty_cache()
        gc.collect()


        return unlabel_idxs[indsAll]