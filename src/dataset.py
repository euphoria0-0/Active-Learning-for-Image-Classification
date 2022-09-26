from email.mime import base
import os
from scipy.io import loadmat
import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as D


dataset_name = {'cifar10':'CIFAR10', 'cifar100':'CIFAR100', 'fashionmnist': 'FashionMNIST', 'caltech256': 'Caltech256'}
dataset_num_classes = {'CIFAR10':10, 'CIFAR100':100, 'FashionMNIST': 10, 'Caltech256': 257}



# image dataset
class ImageDataset(object):
    def __init__(self, data='CIFAR10', data_dir='./dataset/', resize=32, transform=False, pretrained=False):
        self.data = dataset_name[data.lower()]
        self.nClass = dataset_num_classes[self.data]
        self.data_dir = data_dir
        self.pretrained = pretrained
        self.img_size = resize
        self.transform = transform

        if self.data == 'CIFAR10':
            self.nTrain, self.nTest = 50000, 10000

        elif self.data == 'CIFAR100':
            self.nTrain, self.nTest = 50000, 10000

        elif self.data == 'FashionMNIST':
            self.nTrain, self.nTest = 60000, 10000

        elif self.data == 'Caltech256':
            self.nTrain, self.nTest = 22897, 7710

        # get data
        self.dataset = self._getData()

        for data_key in ['train', 'unlabeled', 'test']:
            self.dataset[data_key] = DatasetHandlerIdx(self.dataset[data_key])

    
    def _getData(self):
        dataset = {}
        train_transform, test_transform = self._data_transform()
        
        if self.data in ['CIFAR10', 'CIFAR100', 'FashionMNIST']:
            get_class = getattr(D, self.data)
            dataset['train'] = get_class(self.data_dir + self.data, train=True, download=True, transform=train_transform)
            dataset['unlabeled'] = get_class(self.data_dir + self.data, train=True, download=True, transform=test_transform)
            dataset['test'] = get_class(self.data_dir + self.data, train=False, download=True, transform=test_transform)

        elif self.data == 'Caltech256':
            train_idx_file = os.path.join(self.data_dir, 'ExtractedFeatures/Caltech256_train_idx.txt')
            test_idx_file = os.path.join(self.data_dir, 'ExtractedFeatures/Caltech256_test_idx.txt')

            assert os.path.exists(train_idx_file)

            with open(train_idx_file, 'r') as f:
                train_indices = f.readlines()
            train_indices = list(map(int, train_indices[0].split(',')[:-1]))
            with open(test_idx_file, 'r') as f:
                test_indices = f.readlines()
            test_indices = list(map(int, test_indices[0].split(',')[:-1]))
            train_transform, test_transform = self._data_transform(False)

            train_data = D.ImageFolder(root=self.data_dir+'Caltech256', transform=train_transform)
            test_data = D.ImageFolder(root=self.data_dir+'Caltech256', transform=test_transform)

            dataset['train'] = Subset(train_data, train_indices)
            dataset['unlabeled'] = Subset(test_data, train_indices)
            dataset['test'] = Subset(test_data, test_indices)
        
        return dataset


    def _data_transform(self, random=False):
        if self.data == 'CIFAR10':
            mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
            # [0.4914, 0.4822, 0.4465], [0.2413, 0.2378, 0.2564] # 224x224
            add_transform = []
            if self.transform:
                add_transform.extend([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(size=32, padding=4),
                ])
            
        elif self.data == 'CIFAR100':
            add_transform = []
            if self.transform:
                add_transform.extend([
                    T.RandomHorizontalFlip(),
                    T.RandomCrop(size=32, padding=4)
                ])

            if self.img_size == 32:
                mean, std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]  # original 32x32
            elif self.img_size == 224:
                mean, std =  [0.5071, 0.4865, 0.4409], [0.2623, 0.2513, 0.2714]  # 224x224
                add_transform.append( T.Resize((self.img_size, self.img_size)) )
            else:
                # mean, std = get_img_mean(self.data, self.img_size)
                raise('Select image size from {32, 224}')
            

        elif self.data == 'FashionMNIST':
            mean, std = [0.1307], [0.3081]
            add_transform = [T.Pad(2)]

        elif self.data == 'Caltech256':
            mean, std = [0.5511, 0.5335, 0.5052], [0.3151, 0.3116, 0.3257]
            #add_transform = [T.Resize((224,224))]
            add_transform = [T.Resize((32,32))]

        
        base_transform = [T.ToTensor(), T.Normalize(mean, std)]
        if self.data == 'FashionMNIST':
            base_transform = [T.Grayscale(num_output_channels=1)] + base_transform
        transform = add_transform + base_transform
        test_transform = T.Compose(base_transform)
        train_transform = T.Compose(transform)
        # in this setting, we don't use any data augmentation / transformation methods
        return train_transform, test_transform


# dataset handler for output index of an image from the dataset
class DatasetHandlerIdx(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)


# mean and standard deviation of image dataset
def get_img_mean(dataset_name, img_size):
    transform = T.Compose([T.Resize((img_size,img_size)), T.ToTensor()])
    if dataset_name == 'CIFAR10':
        dataset = D.CIFAR10('../dataset/cifar10', train=True, download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = D.CIFAR100('../dataset/cifar100', train=True, download=True, transform=transform)
    elif dataset_name == 'FashionMNIST':
        # dataset = D.FashionMNIST('../dataset/fashionmnist', train=True, download=True, transform=transform)
        dataset = D.DatasetFolder('../FashionMNIST/TrnImg/TrnImg', transform=transform)
        transform = T.Compose([T.ToTensor()])

    loader = DataLoader(
        dataset,
        batch_size=1000, num_workers=0, shuffle=False
    )

    mean = torch.zeros(3)
    mean2 = torch.zeros(3)
    total = torch.zeros(1)
    print('--> get mean&stdv of images')
    for data, _ in loader:
        mean += torch.sum(data, dim=(0, 2, 3), keepdim=False)
        mean2 += torch.sum((data ** 2), dim=(0, 2, 3), keepdim=False)
        total += data.size(0)

    total *= (data.size(2) ** 2)
    mean /= total
    std = torch.sqrt((mean2 - total * (mean ** 2)) / (total - 1))

    mean = list(np.around(mean.numpy(), 4))
    std = list(np.around(std.numpy(), 4))
    return mean, std