'''
Reference:
	https://github.com/Mephisto405/Learning-Loss-for-Active-Learning
'''
from .activelearner import ActiveLearner

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset


class LearningLoss(ActiveLearner):
	def __init__(self, dataset, args):
		super().__init__(dataset, args)
		self.subset_size = args.subset_size

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
		model['module'].eval()

		# get uncertainty (predicted loss)
		uncertainty = torch.tensor([])
		with torch.no_grad():
			for inputs, _, _ in unlabel_loader:
				inputs = inputs.to(self.device)

				_ = model['backbone'](inputs)
				features = model['backbone'].get_features()
				features = [F.relu(ftr) for ftr in features]  ## added
				pred_loss = model['module'](features)
				pred_loss = pred_loss.view(pred_loss.size(0))
				uncertainty = torch.cat((uncertainty, pred_loss.detach().cpu().data), 0)

				torch.cuda.empty_cache()

		query_indices = np.take(subset, torch.argsort(uncertainty)[-nQuery:].numpy())

		return query_indices
