'''
Reference:
    https://github.com/JordanAsh/badge
'''
from .activelearner import ActiveLearner

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class CoreSet(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.nTrain = args.nTrain

    def query(self, nQuery, model, idxs_lb=None):
        # dataloader
        label_idxs = np.where(idxs_lb == True)[0]
        unlabel_idxs = np.where(idxs_lb == False)[0]

        unlabel_set = Subset(self.dataset['train'], unlabel_idxs)
        
        unlabel_loader = DataLoader(unlabel_set, **self.kwargs)

        model['backbone'].eval()

        # get embedding
        embDim = model['backbone'].get_embedding_dim()
        embedding = np.zeros([self.nTrain, embDim])
        with torch.no_grad():
            for input, _, idxs in unlabel_loader:
                input = input.to(self.device)
                _ = model['backbone'](input)
                e1 = model['backbone'].get_embedding()
                embedding[idxs] = e1.data.cpu().data.numpy()

        # furthest_first
        X = embedding[unlabel_idxs, :]
        X_set = embedding[label_idxs, :]

        m = np.shape(X)[0]
        if np.shape(X_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_set)
            min_dist = np.amin(dist_ctr, axis=1)

        query_indices = []
        for _ in tqdm(range(nQuery), desc='CoreSet', ncols=100):
            idx = min_dist.argmax()
            query_indices.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        # update query data
        query_indices = np.take(unlabel_idxs, query_indices)

        return query_indices



'''
"""
Reference:
    https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
"""
def get_kcg(model, labeled_data_size, unlabeled_loader, args):
    model.eval()
    features = torch.tensor([]).to(args.device)

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.to(args.device)
            _, features_batch, _ = model(inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(args.subset, (args.subset + labeled_data_size))
        sampling = kCenterGreedy(feat)
        batch = sampling.select_batch_(new_av_idx, args.addendum)
        #other_idx = [x for x in range(args.subset) if x not in batch]
    return batch #other_idx + batch
'''

'''
"""
Reference:
    https://github.com/ej0cl6/deep-active-learning
"""
    def query()
        labeled_idxs, train_data = self.dataset.get_train_data()
        embeddings = self.get_embeddings(train_data)
        embeddings = embeddings.numpy()

        dist_mat = np.matmul(embeddings, embeddings.transpose())
        sq = np.array(dist_mat.diagonal()).reshape(len(labeled_idxs), 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.transpose()
        dist_mat = np.sqrt(dist_mat)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]

        for i in tqdm(range(n), ncols=100):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(self.dataset.n_pool)[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)
            
        return np.arange(self.dataset.n_pool)[(self.dataset.labeled_idxs ^ labeled_idxs)]
'''