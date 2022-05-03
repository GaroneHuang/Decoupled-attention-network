from torch.utils.data import Dataset, DataLoader
import time
import math
import numpy as np
import os
import torch
import cv2
import Augment
import random

class LineGenerate():
    def __init__(self, IAMPath, conH, conW, augment = False, training=False):
        self.training = training
        self.augment = augment
        self.conH = conH
        self.conW = conW
        standard = []
        with open(IAMPath) as f:
            for line in f.readlines():
                standard.append(line.strip('\n'))
        self.image = []
        self.label = []
        line_prefix = '/'.join(IAMPath.split('/')[:-1]) + '/lines'
        IAMLine = line_prefix + '.txt'
        count = 0
        with open(IAMLine) as f:
            for line in f.readlines():
                elements = line.split()
                pth_ele = elements[0].split('-')
                line_tag = '%s-%s' % (pth_ele[0], pth_ele[1])
                if line_tag in standard:
                    pth = line_prefix + '/%s/%s-%s/%s.png' % (pth_ele[0], pth_ele[0], pth_ele[1], elements[0])
                    img= cv2.imread(pth, 0) #see channel and type
                    self.image.append(img)
                    self.label.append(elements[-1])
                    count += 1
        self.len = count

    def get_len(self):
        return self.len

    def generate_line(self, idx):
        if self.training:
            image = self.image[idx]
            label = self.label[idx]
        else:
            image = self.image[idx]
            label = self.label[idx]

        h,w = image.shape
        imageN = np.ones((self.conH,self.conW))*255
        beginH = int(abs(self.conH-h)/2)
        beginW = int(abs(self.conW-w)/2)
        if h <= self.conH and w <= self.conW:
            imageN[beginH:beginH+h, beginW:beginW+w] = image
        elif float(h) / w > float(self.conH) / self.conW:
            newW = int(w * self.conH / float(h))
            beginW = int(abs(self.conW-newW)/2)
            image = cv2.resize(image, (newW, self.conH))
            imageN[:,beginW:beginW+newW] = image
        elif float(h) / w <= float(self.conH) / self.conW:
            newH = int(h * self.conW / float(w))
            beginH = int(abs(self.conH-newH)/2)
            image = cv2.resize(image, (self.conW, newH))
            imageN[beginH:beginH+newH] = image
        label = self.label[idx]

        if self.augment and self.training:
            imageN = imageN.astype('uint8')
            if torch.rand(1) < 0.3:
                imageN = Augment.GenerateDistort(imageN, random.randint(3, 8))
            if torch.rand(1) < 0.3:
                imageN = Augment.GenerateStretch(imageN, random.randint(3, 8))
            if torch.rand(1) < 0.3:
                imageN = Augment.GeneratePerspective(imageN)
            
        imageN = imageN.astype('float32')
        imageN = (imageN-127.5)/127.5
        return imageN, label

class WordGenerate():
    def __init__(self, IAMPath, conH, conW, augment = False):
        self.augment = augment
        self.conH = conH
        self.conW = conW
        standard = []
        with open(IAMPath) as f:
            for line in f.readlines():
                standard.append(line.strip('\n'))
        self.image = []
        self.label = []

        word_prefix = '/'.join(IAMPath.split('/')[:-1]) + '/words'
        IAMWord = word_prefix + '.txt'
        count = 0
        with open(IAMWord) as f:
            for line in f.readlines():
                elements = line.split()
                pth_ele = elements[0].split('-')
                line_tag = '%s-%s' % (pth_ele[0], pth_ele[1])
                if line_tag in standard:
                    pth = word_prefix + '/%s/%s-%s/%s.png' % (pth_ele[0], pth_ele[0], pth_ele[1], elements[0])
                    img= cv2.imread(pth, 0) #see channel and type
                    if img is not None:
                        self.image.append(img)
                        self.label.append(elements[-1])
                        count += 1
                    else:
                        print('error')
                        continue;

        self.len = count

    def get_len(self):
        return self.len

    def word_generate(self, index):

        endW = random.randint(0, 49);
        label = ''
        imageN = np.ones((self.conH,self.conW))*255
        imageList =[]
        while True:
            idx = random.randint(0, self.len - 1)
            image = self.image[idx]
            h,w = image.shape
            beginH = int(abs(self.conH-h)/2)
            imageList.append(image)
            if endW + w > self.conW:
                break;
            if h <= self.conH: 
                imageN[beginH:beginH+h, endW:endW+w] = image
            else: 
                imageN[:,endW:endW+w] = image[beginH:beginH+self.conH]

            endW += random.randint(0, 59)+20+w
            if label == '':
                label = self.label[idx]
            else:
                label = label + '|' + self.label[idx]

        label = label
        imageN = imageN.astype('uint8')
        if self.augment:
            if torch.rand(1) < 0.3:
                imageN = Augment.GenerateDistort(imageN, random.randint(3, 8))
            if torch.rand(1) < 0.3:
                imageN = Augment.GenerateStretch(imageN, random.randint(3, 8))
            if torch.rand(1) < 0.3:
                imageN = Augment.GeneratePerspective(imageN)

        imageN = imageN.astype('float32')
        imageN = (imageN-127.5)/127.5
        return imageN, label        

class IAMDataset(Dataset):
    def __init__(self, img_list, img_height, img_width, augment = False, transform=None):
        self.training = True
        self.augment = augment
        IAMPath = img_list
        self.conH = img_height
        self.conW = img_width
        self.LG = LineGenerate(IAMPath, self.conH, self.conW, self.augment, self.training)

    def __len__(self):
        return self.LG.get_len()

    def __getitem__(self, idx):
        
        imageN, label = self.LG.generate_line(idx)

        imageN = imageN.reshape(1,self.conH,self.conW)
        sample = {'image': torch.from_numpy(imageN), 'label': label}

        return sample  

class IAMSynthesisDataset(Dataset):
    def __init__(self, img_list, img_height, img_width, augment = False, transform=None):
        self.training = True
        self.augment = augment
        IAMPath = img_list
        self.conH = img_height
        self.conW = img_width
        self.LG = LineGenerate(IAMPath, self.conH, self.conW, self.augment, self.training)
        self.WG = WordGenerate(IAMPath, self.conH, self.conW, self.augment)

    def __len__(self):
        return self.LG.get_len()

    def __getitem__(self, idx):
        if random.random() < 0.5:
            imageN, label = self.LG.generate_line(idx)
        else:
            imageN, label = self.WG.word_generate(idx)

        imageN = imageN.reshape(1,self.conH,self.conW)
        sample = {'image': torch.from_numpy(imageN), 'label': label}
        return sample

class MixDataset(Dataset):
    def __init__(self, datasets, cfgs, probs):
        super().__init__()
        self.datasets = [dataset(cfg) for dataset, cfg in zip(datasets, cfgs)]
        self.length = np.sum([len(dataset) * prob for dataset, prob in zip(self.datasets, probs)]) * len(datasets)
        self.prob_sums = [0.]
        for prob in probs:
            self.prob_sums.append(self.prob_sums[-1] + prob)
        self.prob_sums = np.array(self.prob_sums[1:])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        rand_num = random.random()
        dataset_idx = np.where(self.prob_sums >= rand_num)[0][0]
        idx = random.randint(0, len(self.datasets[dataset_idx]) - 1)
        sample = self.datasets[dataset_idx][idx]
        return sample
