# coding:utf-8
import torch
import torch.optim as optim
import os
from dataset_hw import *
from DAN import *

global_cfgs = {
    'state': 'Train',
    'epoch': 100,
    'show_interval': 50,
    'test_batch': 1
}

dataset_cfgs = {
    'dataset_train': IAMSynthesisDataset,
    'dataset_train_args': {
        'img_list': 'data/IAM/train_list.txt',
        'img_height': 192,
        'img_width': 2048,
        'augment': True, # with the data augmentation toolkit
    },
    'dataloader_train': {
        'batch_size': 24,
        'shuffle': True,
        'num_workers': 2,
    },

    'dataset_test': IAMDataset,
    'dataset_test_args': {
        'img_list': 'data/IAM/eval_list.txt',
        'img_height': 192,
        'img_width': 2048,
    },
    'dataloader_test': {
        'batch_size': 12,
        'shuffle': False,
        'num_workers': 2,
    },

    'case_sensitive': True,
    'dict_dir': 'dict/dic_79.txt'
}

net_cfgs = {
    'FE': Feature_Extractor,
    'FE_args': {
        'strides': [(2,2), (2,2), (2,1), (2,2), (2,2), (2,1)],
        'compress_layer' : True, 
        'input_shape': [1, 192, 2048], # C x H x W
    },
    'CAM': CAM_transposed,
    'CAM_args': {
        'maxT': 150, 
        'depth': 14, 
        'num_channels': 128,
    },
    'DTD': DTD,
    'DTD_args': {
        'nclass': 80, # extra 2 classes for Unkonwn and End-token
        'nchannel': 256,
        'dropout': 0.7,
    },

    # 'init_state_dict_fe': 'models/hw/exp1_E99_I2000-2295_M0.pth',
    # 'init_state_dict_cam': 'models/hw/exp1_E99_I2000-2295_M1.pth',
    # 'init_state_dict_dtd': 'models/hw/exp1_E99_I2000-2295_M2.pth',

    'init_state_dict_fe': None,
    'init_state_dict_cam': None,
    'init_state_dict_dtd': None,
}

# optimizer_cfgs = {
#     # optim for FE
#     'optimizer_0': optim.SGD,
#     'optimizer_0_args':{
#         'lr': 0.1,
#         'momentum': 0.9,
#     },

#     'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
#     'optimizer_0_scheduler_args': {
#         'milestones': [20, 40, 60, 80],
#         'gamma': 0.3162,
#     },

#     # optim for CAM
#     'optimizer_1': optim.SGD,
#     'optimizer_1_args':{
#         'lr': 0.1,
#         'momentum': 0.9,
#     },
#     'optimizer_1_scheduler': optim.lr_scheduler.MultiStepLR,
#     'optimizer_1_scheduler_args': {
#         'milestones': [20, 40, 60, 80],
#         'gamma': 0.3162,
#     },

#     # optim for DTD
#     'optimizer_2': optim.SGD,
#     'optimizer_2_args':{
#         'lr': 0.1,
#         'momentum': 0.9,
#     },
#     'optimizer_2_scheduler': optim.lr_scheduler.MultiStepLR,
#     'optimizer_2_scheduler_args': {
#         'milestones': [20, 40, 60, 80],
#         'gamma': 0.3162,
#     },
# }

optimizer_cfgs = {
    # optim for FE
    'optimizer_0': optim.Adadelta,
    'optimizer_0_args':{
        'lr': 1.,
        # 'momentum': 0.9,
    },

    'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_0_scheduler_args': {
        'milestones': [],
        'gamma': 0.3162,
    },

    # optim for CAM
    'optimizer_1': optim.Adadelta,
    'optimizer_1_args':{
        'lr': 1.,
        # 'momentum': 0.9,
    },
    'optimizer_1_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_1_scheduler_args': {
        'milestones': [],
        'gamma': 0.3162,
    },

    # optim for DTD
    'optimizer_2': optim.Adadelta,
    'optimizer_2_args':{
        'lr': 1.,
        # 'momentum': 0.9,
    },
    'optimizer_2_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_2_scheduler_args': {
        'milestones': [],
        'gamma': 0.3162,
    },
}

saving_cfgs = {
    'saving_epoch_interval': 1,
    'saving_path': 'models/hw/iam_words2line_aug/',
}

# def mkdir(path_):
#     paths = path_.split('/')
#     command_str = 'mkdir '
#     for i in range(0, len(paths) - 1):
#         command_str = command_str + paths[i] + '/'
#     command_str = command_str[0:-1]
#     os.system(command_str)

def showcfgs(s):
    for key in s.keys():
        print(key , s[key])
    print('')

# mkdir(saving_cfgs['saving_path'])
if not os.path.exists(saving_cfgs['saving_path']):
    os.makedirs(saving_cfgs['saving_path'])