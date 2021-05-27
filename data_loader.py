import os, random
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T


class CelebA(data.Dataset):
    def __init__(self, mode= 'train', transform= None):
        self.imgs_path= 'D:\Downloaded_Datas\img_align_celeba\img_align_celeba'
        self.attrs_path= 'D:\Downloaded_Datas\list_attr_celeba\list_attr_celeba.csv'
        self.selected_attrs= ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
        self.attr2idx= {}
        self.idx2attr= {}
        self.train_dataset= []
        self.test_dataset= []
        self.transform= transform


        lines= [line.rstrip() for line in open(self.attrs_path, 'r')]

        all_attr_names= lines[0].split(',')
        all_attr_names= all_attr_names[1:]
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name]= i
            self.idx2attr[i]= attr_name

        lines= lines[1:]
        random.seed(23)
        random.shuffle(lines)

        for i, line in enumerate(lines):
            split= line.split(',')
            filename= split[0]
            values= split[1:]
            label= []
            for attr_name in self.selected_attrs:
                idx= self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+ 1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])
        
        self.dataset= self.train_dataset if mode == 'train' else self.test_dataset
        self.num_imgs= len(self.dataset)

    
    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        filename, label= self.dataset[idx]
        img= Image.open(os.path.join(self.imgs_path, filename))
        return self.transform(img), torch.FloatTensor(label)
    



def CelebA_Loader(mode= 'train'):
    transform= []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(178))
    transform.append(T.Resize(128))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean= (0.5, 0.5, 0.5), std= (0.5, 0.5, 0.5)))

    transforms= T.Compose(transform)

    dataset_celeba= CelebA(mode= mode, transform= transforms)

    loader= data.DataLoader(dataset= dataset_celeba,
    batch_size= 16,
    shuffle= (mode=='train'))
    
    return loader