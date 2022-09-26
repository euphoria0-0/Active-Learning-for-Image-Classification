'''
Reference:
    https://github.com/cubeyoung/TA-VAAL
'''
from .activelearner import ActiveLearner
from ..model import VAE, Discriminator

import sys
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Subset, Dataset


class TAVAAL(ActiveLearner):
    def __init__(self, dataset, args):
        super().__init__(dataset, args)
        self.subset_size = args.subset_size
        self.batch_size = args.batch_size
        self.num_epoch_vaal = args.num_epoch_vaal

        self.train_set = self.dataset['train']
        self.test_set = self.dataset['test']

        self.beta = args.weight
        self.adversary_param = 10 if args.data.lower() == 'caltech256' else 1
        self.num_vae_steps = 1
        self.num_adv_steps = 1
        self.vaal_lr = 5e-4
        self.nChannel = 1 if args.data.lower() == 'fashionmnist' else 3
        self.latent_dim = 64 if args.data.lower() == 'caltech256' else 32


    def query(self, nQuery, model, idxs_lb=None):
        '''
        query data
        '''
        # dataloader
        unlabel_idxs = np.where(idxs_lb == False)[0]
        if self.subset_size is None or self.subset_size > len(unlabel_idxs):
            subset = unlabel_idxs
        else:
            subset = np.random.choice(unlabel_idxs, self.subset_size, replace=False)

        # train VAE and Discriminator
        model = self.train_tavaal(idxs_lb, subset, model)
        
        # get dataloader
        np.random.shuffle(subset)
        unlabel_set = Subset(self.train_set, subset)

        unlabel_loader = DataLoader(unlabel_set, **self.kwargs)
        
        # get predictions
        model['backbone'].eval()
        model['vae'].eval()
        model['discriminator'].eval()
        model['module'].eval()

        all_preds = []
        for data in tqdm(unlabel_loader, desc='>> query', ncols=100):
            input = data[0]
            input = input.to(self.device)
            
            with torch.no_grad():
                _ = model['backbone'](input)
                features = model['backbone'].get_features()
                r = model['module'](features)
                _, _, mu, _ = model['vae'](input, r=torch.sigmoid(r))
                preds = model['discriminator'](mu, r=r)

            preds = preds.cpu().data
            all_preds.extend(preds)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, query_indices = torch.topk(all_preds, nQuery)

        query_indices = np.take(subset, query_indices)

        return query_indices
    
    
    def train_tavaal(self, idxs_lb, subset, models):

        num_iter = (len(idxs_lb) * self.num_epoch_vaal) // self.batch_size
        # num_iter = 10  # just for debugging

        # load data
        labeled_idxs = np.where(idxs_lb == True)[0]
        label_loader = DataLoader(Subset(self.train_set, labeled_idxs), **self.kwargs)
        unlabel_loader = DataLoader(Subset(self.train_set, subset), **self.kwargs)

        
        labeled_data = read_img_data(label_loader)
        unlabeled_data = read_img_data(unlabel_loader)

        # load model
        vae = VAE(z_dim=self.latent_dim, nc=self.nChannel, tavaal=True).to(self.device)
        discriminator = Discriminator(z_dim=self.latent_dim, tavaal=True).to(self.device)

        models['backbone'].eval()
        models['module'].eval()
        vae.train()
        discriminator.train()

        vae = vae.to(self.device)
        discriminator = discriminator.to(self.device)
        models['backbone'] = models['backbone'].to(self.device)
        models['module'] = models['module'].to(self.device)

        # loss
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        # optimizers
        optim_vae = optim.Adam(vae.parameters(), self.vaal_lr)
        optim_discriminator = optim.Adam(discriminator.parameters(), self.vaal_lr)
        optimizers = {'vae': optim_vae, 'discriminator': optim_discriminator}

        ## training
        for iter in tqdm(range(num_iter), desc='>> TAVAAL training', ncols=100):

            # load data
            labeled_imgs = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)
            
            labeled_imgs = labeled_imgs.to(self.device)
            unlabeled_imgs = unlabeled_imgs.to(self.device)

            ## get features and ranks
            if iter == 0:
                r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0], 1))).type(torch.FloatTensor).to(self.device)
                r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0], 1))).type(torch.FloatTensor).to(self.device)

                r_l = r_l_0.detach()
                r_u = r_u_0.detach()
                r_l_s = r_l_0.detach()
                r_u_s = r_u_0.detach()
            else:
                with torch.no_grad():
                    _ = models['backbone'](labeled_imgs)
                    features_l = models['backbone'].get_features()
                    _ = models['backbone'](unlabeled_imgs)
                    features_u = models['backbone'].get_features()

                    r_l = models['module'](features_l)
                    r_u = models['module'](features_u)

                r_l_s = torch.sigmoid(r_l).detach()
                r_u_s = torch.sigmoid(r_u).detach()

            ## training VAE model
            for vae_step in range(self.num_vae_steps):
                # labeled set VAE
                recon, _, mu, logvar = vae(labeled_imgs, r=r_l_s)
                unsup_loss = vae_loss(mse_loss(recon, labeled_imgs), mu, logvar, self.beta)

                # unlabeled set VAE
                unlab_recon, _, unlab_mu, unlab_logvar = vae(unlabeled_imgs, r=r_u_s)
                transductive_loss = vae_loss(mse_loss(unlab_recon, unlabeled_imgs), unlab_mu, unlab_logvar, self.beta)

                # discriminator
                labeled_preds = discriminator(mu, r=r_l)
                unlabeled_preds = discriminator(unlab_mu, r=r_u)

                lab_real_preds = torch.ones(labeled_imgs.size(0)).to(self.device)
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0)).to(self.device)

                dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                           bce_loss(unlabeled_preds[:,0], unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.adversary_param * dsc_loss

                optimizers['vae'].zero_grad()
                total_vae_loss.backward()
                optimizers['vae'].step()

                del unsup_loss
                del transductive_loss
                del dsc_loss

                # sample new batch if needed to train the adversarial network
                if vae_step < (self.num_vae_steps - 1):
                    labeled_imgs = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    labeled_imgs = labeled_imgs.to(self.device)
                    unlabeled_imgs = unlabeled_imgs.to(self.device)

            ## training Discriminator model
            for adv_step in range(self.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs, r=r_l_s)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs, r=r_u_s)

                labeled_preds = discriminator(mu, r=r_l)
                unlabeled_preds = discriminator(unlab_mu, r=r_u)

                lab_real_preds = torch.ones(labeled_imgs.size(0)).to(self.device)
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)).to(self.device)

                dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + \
                           bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

                optimizers['discriminator'].zero_grad()
                dsc_loss.backward()
                optimizers['discriminator'].step()

                # sample new batch if needed to train the adversarial network
                if adv_step < (self.num_adv_steps-1):
                    labeled_imgs = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    labeled_imgs = labeled_imgs.to(self.device)
                    unlabeled_imgs = unlabeled_imgs.to(self.device)

        sys.stdout.write(
            '\rIter {}/{} VAELoss {:.6f} AdvLoss {:.6f}'.format(iter + 1, num_iter, total_vae_loss, dsc_loss))
        sys.stdout.flush()

        models['vae'] = vae
        models['discriminator'] = discriminator

        return models




def read_img_data(dataloader):
    while True:
        for img, _, _ in dataloader:
            if img is None:
                sys.exit(0)
            yield img


def vae_loss(MSE, mu, logvar, beta):
    #MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


class CombineDatasetHandler(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        assert len(self.dataset1) == len(self.dataset2)

    def __getitem__(self, index):
        input1, _, idx1 = self.dataset1[index]
        input2, _, idx2 = self.dataset2[index]
        assert idx1 == idx2
        return input1, input2, index

    def __len__(self):
        return len(self.dataset1)