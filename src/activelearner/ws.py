'''
Reference:
    Weight Decay Scheduling and Knowledge Distillation for Active Learning
    https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710426.pdf
'''
from .activelearner import ActiveLearner

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class WeightDecayScheduling(ActiveLearner):
	def __init__(self, dataset, args):
		super().__init__(dataset, args)
		# hyperparameters
		self.sampling_type = args.ws_sampling_type


	def query(self, nQuery, model, idxs_lb=None):
		'''
		query data
		'''
		unlabeled_idx = np.where(idxs_lb==False)[0]

		train_set = self.dataset['train']
		all_train_loader = DataLoader(train_set, **self.kwargs)
		model['backbone'].eval()

		# inference for entropy
		entropy_total = []
		with torch.no_grad():
			for inputs, _, _ in all_train_loader:
				inputs = inputs.to(self.device)

				# compute output
				outputs = model['backbone'](inputs)

				# compute entropy for active learning
				outputs = outputs.detach().data
				entropy = F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)
				entropy_total.extend(-1.0 * entropy.sum(dim=1))

		entropy = torch.stack(entropy_total).cpu()

		# query data
		if self.sampling_type == 'descending':
			query_indices = unlabeled_idx[np.argsort(-entropy[unlabeled_idx])][:nQuery]
		elif self.sampling_type == 'ascending':
			query_indices = unlabeled_idx[np.argsort(entropy[unlabeled_idx])][:nQuery]
		elif self.sampling_type == 'sampled_descending':
			np.random.shuffle(unlabeled_idx)
			unlabeled_idx = unlabeled_idx[:10000]
			query_indices = unlabeled_idx[np.argsort(-entropy[unlabeled_idx])][:nQuery]

		return query_indices