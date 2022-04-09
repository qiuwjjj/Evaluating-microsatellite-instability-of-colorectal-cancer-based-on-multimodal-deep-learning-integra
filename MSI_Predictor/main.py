from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.common import logger
from utils.custom_dset import CustomDset
# from utils.analytics import draw_roc, draw_roc_for_multiclass

import train_test_split_
from train_ import train_model
from test_ import test

from Net_ import Net, Cnn_With_Clinical_Net

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def generative_model(model, clinical=False):  # 生成式模型
    image_datasets = {x: CustomDset(os.getcwd()+f'/data/269image/train.csv', data_transforms[x]) for x in ['train']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                        shuffle =True, num_workers=4) for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    class_names = image_datasets['train'].classes

    logger.info(f'model {model} / train')

    available_policies = {"resnet18": models.resnet18, "vgg16": models.vgg16, "vgg19": models.vgg19, "resnet101":models.resnet101
            , "inception": models.inception_v3}
    
    model_ft = available_policies[model](pretrained=True)
    
    if clinical:
        model_ft = Cnn_With_Clinical_Net(model_ft)
    else:
        model_ft = Net(model_ft)
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized 观察到所有的参数都在进行优化
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    # Decay LR by a factor of 0.1 every 7 epochs 每7个epochs就衰减学习率为0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft, tb = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, 
        dataset_sizes, num_epochs=25, clinical=clinical)
    tb.close()

    save_model = os.getcwd()+f'/results/269image/models/{model}'
    if clinical:
        save_model = save_model + '_clinical'
    save_model = save_model + '.pkl'
    
    torch.save(model_ft, save_model)


def main(ocs, classification, clinical):
     
    # train_test_splitter.main("/media/zw/Elements1/tiles_cn", "/home/xisx/tmbpredictor/labels/uteri.csv")
    generative_model("resnet101",  clinical=clinical)  #训练
    测试
    path = os.getcwd()+f'/results/269image/models/resnet101'
    if clinical:
        path = path + '_clinical'
    model_ft = torch.load(path + '.pkl')
    test(model_ft, "resnet18")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--classification', type=int, default=2)
    parser.add_argument('--clinical', type=bool, default=False)
    args = parser.parse_args()

    origirn_classfication_set = None

    main(origirn_classfication_set, args.classification, args.clinical)