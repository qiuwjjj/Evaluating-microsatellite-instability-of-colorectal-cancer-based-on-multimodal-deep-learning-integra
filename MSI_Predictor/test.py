import torch
import numpy as np
from torchvision import transforms
from torch.nn import functional as F

import os
from utils.custom_dset import CustomDset
from utils.common import logger
import json
from sklearn import preprocessing

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def test(model, model_name,  types=0, clinical=False):
    model.eval()

    if clinical:
        with open('./clinical.json', encoding='utf-8') as f:
            json_dict = json.load(f)
        peoples = [i for i in json_dict]
        features = np.array([json_dict[i] for i in json_dict], dtype=np.float32)
        min_max_scaler = preprocessing.MinMaxScaler()
        clinical_features = min_max_scaler.fit_transform(features)
    
    testset = CustomDset(os.getcwd()+f'/data/269image/test.csv', data_transforms['test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=4)

    person_prob_dict = dict()
    with torch.no_grad():
        for data in testloader:
            images, labels, names, images_names = data
            if clinical:
                X_train_minmax = [clinical_features[peoples.index(i)] for i in names]
                outputs = model(images.to(device), torch.from_numpy(np.array(X_train_minmax, dtype=np.float32)).to(device))
            else:
                outputs = model(images.to(device))
            probability = F.softmax(outputs, dim=1).data.squeeze()
            probs = probability.cpu().numpy()
            for i in range(labels.size(0)):
                p = names[i]
                if p not in person_prob_dict.keys():
                    person_prob_dict[p] = {
                        'prob_0': 0, 
                        'prob_1': 0,
                        'label': labels[i].item(),
                        'img_num': 0}
                if probs.ndim == 2:
                    person_prob_dict[p]['prob_0'] += probs[i, 0]
                    person_prob_dict[p]['prob_1'] += probs[i, 1]
                    person_prob_dict[p]['img_num'] += 1
                else:
                    person_prob_dict[p]['prob_0'] += probs[0]
                    person_prob_dict[p]['prob_1'] += probs[1]
                    person_prob_dict[p]['img_num'] += 1


    y_true = []
    y_pred = []
    score_list = []

    total = len(person_prob_dict)
    correct = 0
    for key in person_prob_dict.keys():
        predict = 0
        if person_prob_dict[key]['prob_0'] < person_prob_dict[key]['prob_1']:
            predict = 1
        if person_prob_dict[key]['label'] == predict:
            correct += 1
        y_true.append(person_prob_dict[key]['label'])
        score_list.append([person_prob_dict[key]['prob_0']/person_prob_dict[key]["img_num"], person_prob_dict[key]['prob_1']/person_prob_dict[key]["img_num"]])
        y_pred.append(predict)
        open(f'{model_name}_confusion_matrix_classification_{types}.txt', 'a+').write(str(person_prob_dict[key]['label'])+"\t"+str(predict)+'\n')
    
    np.save(os.getcwd()+f'/results/269image/models/y_true_{0}.npy', np.array(y_true))
    np.save(os.getcwd()+f'/results/269image/models/score_{0}.npy', np.array(score_list))
    np.save(os.getcwd()+f'/results/269image/models/y_pred_{0}.npy', np.array(y_pred))
    logger.info('Accuracy of the network on test images: %d %%' % (100 * correct / total))


