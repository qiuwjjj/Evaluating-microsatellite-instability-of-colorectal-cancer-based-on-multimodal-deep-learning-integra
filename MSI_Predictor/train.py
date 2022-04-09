import torch
import time
import copy
from utils.common import logger
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import json
from sklearn import preprocessing


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tb = SummaryWriter('/home/qiuwj/msipredictorr/runs/269image/')


def train_model(model, criterion, optimizer, scheduler,
                dataloaders, dataset_sizes, num_epochs=25, clinical=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if clinical:
        with open('./clinical.json', encoding='utf-8') as f:
            json_dict = json.load(f)
        peoples = [i for i in json_dict]
        features = np.array([json_dict[i] for i in json_dict], dtype=np.float32)
        min_max_scaler = preprocessing.MinMaxScaler()
        clinical_features = min_max_scaler.fit_transform(features)
    
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))

        phase = 'train'
        model.train()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.遍历数据集
        for inputs_, labels_, names_, _ in dataloaders[phase]:
            inputs_ = inputs_.to(device)
            labels_ = labels_.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train 只有在训练集上，才追踪历史
            with torch.set_grad_enabled(phase == 'train'):
                if clinical:
                    X_train_minmax = [clinical_features[peoples.index(i)] for i in names_]
                    outputs_ = model(inputs_, torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
                else:
                    outputs_ = model(inputs_)
                _, preds = torch.max(outputs_, 1)
                loss = criterion(outputs_, labels_)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs_.size(0)
            running_corrects += torch.sum((preds == labels_.data).int())

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects / dataset_sizes[phase]

        scheduler.step()
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        logger.info('{} Loss: {:.3f} Acc: {:.3f}'.format(
            phase, epoch_loss, epoch_acc))
        tb.add_scalar("Train/Loss", epoch_loss, epoch)
        tb.add_scalar("Train/Accuracy", epoch_acc, epoch)
        tb.flush()

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.3f}h {:.3f}m'.format(
        time_elapsed // 3600, (time_elapsed-time_elapsed // 3600) * 60))
    logger.info('Best train Acc: {:3f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, tb
