'''
Reference:
    https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
'''
from .activelearner import ActiveLearner
from ..model import GCN
from .kcenterGreedy import kCenterGreedy

from tqdm import tqdm
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Subset


class SequentialGCN(ActiveLearner):
    def __init__(self, dataset, args, uncertain_gcn=True):
        super().__init__(dataset, args)
        # hyperparameters
        self.subset_size = args.subset_size

        # optimizer
        self.lr = args.lr_gcn
        self.wdecay = args.wdecay

        # SeqGCN
        self.hidden_units = args.hidden_units
        self.dropout_rate = args.dropout_rate
        self.lambda_loss = args.lambda_loss
        self.s_margin = args.s_margin
        self.num_epoch_gcn = 200

        self.uncertain_gcn = uncertain_gcn


    def query(self, nQuery, model, idxs_lb=None):
        '''
        query data
        '''
        print('>> SeqGCN')
        # dataloader
        labeled_idxs = np.where(idxs_lb == True)[0]
        unlabeled_idxs = np.where(idxs_lb == False)[0]
        if self.subset_size is None or self.subset_size > len(unlabeled_idxs):
            subset = unlabeled_idxs.tolist()
        else:
            subset = np.random.choice(unlabeled_idxs, self.subset_size, replace=False).tolist()

        # Create unlabeled dataloader for the unlabeled subset
        indices = subset + labeled_idxs.tolist()
        unlabeled_set = Subset(self.dataset['train'], indices)
        unlabeled_loader = DataLoader(unlabeled_set, **self.kwargs)

        #binary_labels = torch.cat((torch.zeros([self.subset_size, 1]), (torch.ones([len(labeled_idxs), 1]))), 0)

        # get features
        features = get_features(model['backbone'], unlabeled_loader, self.device)
        features = F.normalize(features)
        adj = aff_to_adj(features, device=self.device)

        # GCN
        gcn_module = GCN(nfeat=features.shape[1],
                         nhid=self.hidden_units,
                         nclass=1,
                         dropout=self.dropout_rate).to(self.device)

        gcn_optimizer = optim.Adam(gcn_module.parameters(), lr=self.lr, weight_decay=self.wdecay)

        #lbl = np.arange(self.subset_size, max(self.subset_size + len(labeled_idxs), len(self.dataset['train'][1])), 1)
        lbl = np.arange(len(subset), len(subset) + len(labeled_idxs), 1)
        nlbl = np.arange(0, len(subset), 1)

        gcn_module.train()  # original code skiped this..

        ############
        features = features.to(self.device)
        for _ in tqdm(range(self.num_epoch_gcn), ncols=80):
            gcn_optimizer.zero_grad()
            outputs, _, _ = gcn_module(features, adj)
            loss = BCEAdjLoss(outputs.cpu(), lbl, nlbl, self.lambda_loss)
            loss.backward()
            gcn_optimizer.step()
            sys.stdout.write('\rGCN Loss {:.6f}'.format(loss.item()))
            sys.stdout.flush()
        print()

        gcn_module.eval()
        with torch.no_grad():
            inputs = features.to(self.device)
            #labels = binary_labels.to(self.device)
            scores, _, feat = gcn_module(inputs, adj)

            if self.uncertain_gcn:
                s_margin = self.s_margin
                scores_median = np.squeeze(torch.abs(scores[:len(subset)] - s_margin).detach().cpu().numpy())
                arg = np.argsort(-(scores_median))[-nQuery:]
                query_indices = np.take(subset, arg)
            else:
                feat = features.detach().cpu().numpy()
                new_av_idx = np.arange(len(subset), (len(subset) + + len(labeled_idxs)))
                sampling2 = kCenterGreedy(feat)
                batch2 = sampling2.select_batch_(new_av_idx, nQuery)
                other_idx = [x for x in range(len(subset)) if x not in batch2]
                arg = other_idx + batch2
                query_indices = arg[-nQuery:]

            '''
            print("Max confidence value: ", torch.max(scores.data))
            print("Mean confidence value: ", torch.mean(scores.data))
            preds = torch.round(scores)
            correct_labeled = (preds[self.subset_size:, 0] == labels[self.subset_size:, 0]).sum().item() / (
                        (cycle + 1) * nQuery)
            correct_unlabeled = (preds[:self.subset_size, 0] == labels[:self.subset_size, 0]).sum().item() / self.subset_size
            correct = (preds[:, 0] == labels[:, 0]).sum().item() / (self.subset_size + (cycle + 1) * nQuery)
            print("Labeled classified: ", correct_labeled)
            print("Unlabeled classified: ", correct_unlabeled)
            print("Total classified: ", correct)
            '''

        return query_indices



def get_features(model, unlabeled_loader, device):
    model.eval()

    features = torch.tensor([]).to(device)
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.to(device)
            _ = model(inputs)
            features_batch = model.get_embedding()
            features_batch = F.relu(features_batch)
            features = torch.cat((features, features_batch), 0)

        feat = features.detach().cpu() #.numpy()
    return feat


def aff_to_adj(x, device='cpu'):
    x = x.detach().to(device)
    adj = torch.matmul(x, torch.t(x))
    adj += -1.0 * torch.eye(adj.shape[0]).to(device)
    adj_diag = torch.sum(adj, axis=0)  # rowise sum
    adj = torch.matmul(adj, torch.diag(1 / adj_diag))
    adj = adj + torch.eye(adj.shape[0]).to(device)

    return adj


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj * unlabeled_score
    return bce_adj_loss