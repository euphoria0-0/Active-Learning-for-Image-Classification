import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

from .model import LossPredLoss


class Trainer:
    def __init__(self, dataset, model, args, writer=None, round=None):
        """
        trainer
        -
        Args:
            dataset: dataset
            model: given model  for training (or test)
            args: arguments for AL training
            writer: tensorboard writer (default: None)
            round: round index (default: None)
        """
        self.train_set = dataset['train']
        self.test_set = dataset['test']
        
        self.device = args.device
        self.writer = writer
        self.round = round  # current round

        # hyperparameter
        self.wdecay = args.wdecay
        if args.al_model == 'ws':  # WeightDecayScheduling method
            self.wdecay /= (round + 1)
        self.momentum = args.momentum
        self.num_epoch = args.num_epoch
        self.subset_size = args.subset_size

        self.kwargs_optim = {'lr': args.lr, 'weight_decay': self.wdecay}
        self.kwargs_scheduler = {'step_size': 10, 'gamma': 0.1}
        self.kwargs_loader = {'batch_size': args.batch_size, 'pin_memory': True, 'shuffle': True}

        # for module
        self.margin = args.margin
        self.weight = args.weight
        self.epoch_loss = args.epoch_loss
        self.flag_train_module = True

        # model
        self.flag_module = args.al_model in ['learningloss', 'tavaal']
        self.flag_vaal = args.al_model in ['vaal', 'tavaal']

        self.model = model
        self.model['backbone'] = self.model['backbone'].to(self.device)
        if self.flag_module:
            self.model['module'] = self.model['module'].to(self.device)

        # loss
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # optimizer
        if args.optimizer == 'sgd':
            optim_backbone = optim.SGD(self.model['backbone'].parameters(), **self.kwargs_optim, momentum=self.momentum)
        else:
            optim_backbone = optim.Adam(self.model['backbone'].parameters(), **self.kwargs_optim)

        if args.lrscheduler == 'step':
            scheduler_backbone = optim.lr_scheduler.StepLR(optim_backbone, **self.kwargs_scheduler)
        elif args.lrscheduler == 'multistep':
            scheduler_backbone = optim.lr_scheduler.MultiStepLR(optim_backbone, milestones=list(map(int, args.milestone.split(','))))

        self.optimizer = {'backbone': optim_backbone}
        self.scheduler = {'backbone': scheduler_backbone}

        if self.flag_module:
            if args.optimizer == 'sgd':
                optim_module = optim.SGD(self.model['module'].parameters(), **self.kwargs_optim, momentum=self.momentum)
            else:
                optim_module = optim.Adam(self.model['module'].parameters(), **self.kwargs_optim)
            
            if args.lrscheduler == 'step':
                scheduler_module = optim.lr_scheduler.StepLR(optim_module, **self.kwargs_scheduler)
            elif args.lrscheduler == 'multistep':
                scheduler_module = optim.lr_scheduler.MultiStepLR(optim_module, milestones=list(map(int, args.milestone.split(','))))

            self.optimizer['module'] = optim_module
            self.scheduler['module'] = scheduler_module


    def train(self, idxs_lb):
        """
        train models
        -
        Args:
            idxs_lb: labeled indices
        Return:
            result: loss, acc
        """
        labeled_idxs = np.where(idxs_lb)[0]
        labeled_set = Subset(self.train_set, labeled_idxs)
        
        dataloader = DataLoader(labeled_set, **self.kwargs_loader)

        self.model['backbone'].train()
        if self.flag_module:
            self.model['module'].train()

        for epoch in tqdm(range(self.num_epoch), ncols=80):

            if epoch > self.epoch_loss:
                self.flag_train_module = False

            result = self._train_epoch(dataloader)

            sys.stdout.write(
                '\rEpoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch + 1, self.num_epoch,
                                                                        result['loss'], result['acc']))
            
            if self.writer is not None:
                self.writer.add_scalar(f'round {self.round} #Labeled {len(labeled_idxs)} Train Loss', result['loss'], epoch)
                self.writer.add_scalar(f'round {self.round} #Labeled {len(labeled_idxs)} Train Acc', result['acc'], epoch)

                # to track the results
                # self.writer.add_scalars(f'round {self.round} #Labeled {len(labeled_idxs)} Loss', {'train':result['loss'], 'test':test_result['loss']}, epoch)
                # self.writer.add_scalars(f'round {self.round} #Labeled {len(labeled_idxs)} Acc', {'train':result['acc'], 'test':test_result['acc']}, epoch)

                if epoch % 10 == 0:
                    self.writer.flush()

            self.scheduler['backbone'].step()
            if self.flag_module:
                self.scheduler['module'].step()

        return result

    def _train_epoch(self, dataloader):
        """
        train one epoch
        -
        Args:
            model: model
            dataloader: dataloader
        Return:
            result: loss, acc
        """
        train_loss, correct, total = 0., 0, 0
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer['backbone'].zero_grad()
            if self.flag_module:
                self.optimizer['module'].zero_grad()

            output = self.model['backbone'](inputs)

            backbone_loss = self.criterion(output, labels.long())

            # train module model if there is
            if self.flag_module:
                features = self.model['backbone'].get_features()
                if not self.flag_train_module:
                    # After certain epochs, stop the gradient from the loss prediction module propagated to the target model.
                    features = [ftr.detach() for ftr in features]

                pred_loss = self.model['module'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))

                module_loss = LossPredLoss(pred_loss, backbone_loss, margin=self.margin)
                backbone_loss = torch.sum(backbone_loss) / backbone_loss.size(0)
                loss = backbone_loss + self.weight * module_loss
            else:
                loss = torch.sum(backbone_loss) / backbone_loss.size(0)

            loss.backward()
            self.optimizer['backbone'].step()
            if self.flag_module:
                self.optimizer['module'].step()
            
            # metric
            _, preds = torch.max(output.data, 1)
            train_loss += torch.sum(backbone_loss.detach()).cpu().item()
            correct += preds.eq(labels).sum().detach().cpu().data.numpy()
            total += inputs.size(0)


        return {'loss': train_loss / total, 'acc': correct / total}

    # @torch.no_grad()
    def test(self, tracking=True):
        """
        test
        -
        Args:
            model: model for test
            data: dataset for test
        Return:
            result: loss, acc
        """
        test_loader = DataLoader(self.test_set, **self.kwargs_loader)

        self.model['backbone'].eval()
        if self.flag_module:
            self.model['module'].eval()

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0

            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                output = self.model['backbone'](inputs)

                loss = self.criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                test_loss += torch.sum(loss).detach().cpu().item()
                correct += preds.eq(labels).sum().detach().cpu().data.numpy()
                total += inputs.size(0)

        assert total > 0

        result = {'loss': test_loss / total, 'acc': correct / total}
        if tracking:
            print(' TestAcc {:.4f}'.format(result['acc']))

        return result

    def get_model(self):
        return self.model
