# -*- coding: UTF-8 -*-

import os
import random
import shutil
from glob import glob
import pandas as pd
import numpy as np
import argparse
import yaml
from utils.common import logger
from pathlib import Path
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

with open(os.path.join(os.getcwd(), 'config/config.yml'), 'r', encoding='utf8') as fs:
    cfg = yaml.load(fs, Loader=yaml.FullLoader)
K = cfg['k']

# available_policies = {'MSI':1, 'MSS':0}
available_policies = {}
# 多阈值
save_paths = {"halves": os.getcwd() + f"/data/269image/"}

def allDataToTrain(X, y, divisionMethod):
    train_data = []
    for (p, label) in zip(X, y):
        for img in glob(p + "/*"):
            train_data.append((img, label))

    pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(
        drop=True)

    # Get the smallest number of image in each category
    min_num = min(pdf['label'].value_counts())

    # Random downsampling
    index = []
    for i in range(2):
        if i == 0:
            start = 0
            end = pdf['label'].value_counts()[i]
        else:
            start = end
            end = end + pdf['label'].value_counts()[i]
        index = index + random.sample(range(start, end), min_num)

    pdf = pdf.iloc[index].reset_index(drop=True)

    # Shuffle
    pdf = pdf.reindex(np.random.permutation(pdf.index)).reset_index(drop=True)
    pdf.to_csv(save_paths[divisionMethod] + f"train.csv", index=None, header=None)

# 交叉验证
def useCrossValidation(X, y, divisionMethod):
    print(divisionMethod)
    skf = StratifiedKFold(n_splits=K, shuffle=True)
    for fold, (train, test) in enumerate(skf.split(X, y)):
        train_data = []
        test_data = []

        train_set, train_label = pd.Series(X).iloc[train].tolist(), pd.Series(y).iloc[train].tolist()
        test_set, test_label = pd.Series(X).iloc[test].tolist(), pd.Series(y).iloc[test].tolist()

        for (data, label) in zip(train_set, train_label):
            # print(data, label)
            for img in glob(data + '/*'):
                train_data.append((img, label))
        for (data, label) in zip(test_set, test_label):
            for img in glob(data + '/*'):
                test_data.append((img, label))

        pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(
            drop=True)

        # Get the smallest number of image in each category
        min_num = min(pdf['label'].value_counts())

        # Random downsampling
        index = []
        for i in range(2):
            if i == 0:
                start = 0
                end = pdf['label'].value_counts()[i]
            else:
                start = end
                end = end + pdf['label'].value_counts()[i]
            index = index + random.sample(range(start, end), min_num)

        pdf = pdf.iloc[index].reset_index(drop=True)

        # Shuffle
        pdf = pdf.reindex(np.random.permutation(pdf.index)).reset_index(drop=True)
        print(save_paths[divisionMethod] + f"train_{fold}.csv")
        pdf.to_csv(save_paths[divisionMethod] + f"train_{fold}.csv", index=None, header=None)

        pdf1 = pd.DataFrame(test_data)
        pdf1.to_csv(save_paths[divisionMethod] + f"test_{fold}.csv", index=None, header=None)

#划分数据集 train：test = 7：3
def split(X, y, divisionMethod, test_size):
    print("here!!!!!!!!")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    train_data = []
    test_data = []
    for (data, label) in zip(X_train, y_train):
        # print(data, label)
        for img in glob(data + '/*'):
            train_data.append((img, label))
    for (data, label) in zip(X_test, y_test):
        for img in glob(data + '/*'):
            test_data.append((img, label))

    pdf = pd.DataFrame(train_data, columns=['img', 'label']).sort_values(by=['label'], ascending=True).reset_index(drop=True)

    # Get the smallest number of image in each category
    min_num = min(pdf['label'].value_counts())

        # Random downsampling
    index = []
    for i in range(2):
        if i == 0:
            start = 0
            end = pdf['label'].value_counts()[i]
        else:
            start = end
            end = end + pdf['label'].value_counts()[i]
        index = index + random.sample(range(start, end), min_num)

    pdf = pdf.iloc[index].reset_index(drop=True)

    # Shuffle
    pdf = pdf.reindex(np.random.permutation(pdf.index)).reset_index(drop=True)
    print(save_paths[divisionMethod] + f"train.csv")
    pdf.to_csv(save_paths[divisionMethod] + f"train.csv", index=None, header=None)

    pdf1 = pd.DataFrame(test_data)
    print(save_paths[divisionMethod] + f"test.csv")
    pdf1.to_csv(save_paths[divisionMethod] + f"test.csv", index=None, header=None)


def main(srcImg, label, divisionMethod, split_type=0, test_size=0.3, shuffle=True):
    assert os.path.exists(label), "Error: 标签文件不存在"
    assert Path(label).suffix == '.csv', "Error: 标签文件需要是csv文件"

    try:
        df = pd.read_csv(label, usecols=["ID", "premsi"])
    except:
        print("Error: 未在文件中发现ID或premsi列信息")

    img_dir = glob(os.path.join(srcImg, '*'))
    xml_file_seq = [img.split('/')[-2] for img in img_dir]

    msi_label_seq = [getattr(row, 'ID') for row in df.itertuples() if getattr(row, 'premsi') == "MSI"]
    mss_label_seq = [getattr(row, 'ID') for row in df.itertuples() if getattr(row, 'premsi') == "MSS"]

    assert msi_label_seq != 0, "Error: 数据分布异常"
    assert mss_label_seq != 0, "Error: 数据分布异常"

    X = []
    y = []



    for msi in msi_label_seq:
        if os.path.join(srcImg, msi) in img_dir:
        # print(os.path.join(srcImg, msi))
            X.append(os.path.join(srcImg, msi))
            y.append(1)

    for mss in mss_label_seq:
        if os.path.join(srcImg, mss) in img_dir:
            X.append(os.path.join(srcImg, mss))
            y.append(0)

    if split_type == 0:
        useCrossValidation(X, y, divisionMethod)  # 交叉验证
    elif split_type == 1:
        allDataToTrain(X, y, divisionMethod)     # 所有数据均训练
    else:
        split(X, y, divisionMethod, test_size)   # train：test = 7：3


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script',
                                     epilog="authorized by geneis ")
    parser.add_argument('--stained_tiles_home', type=str, default="/media/zw/Elements1/qiuwj/269CRC/CRC1/")
    parser.add_argument('--label_dir_path', type=str, default="/home/qiuwj/msipredictor/labels/label269.csv")
    parser.add_argument('--divisionMethod', type=str, default="halves")
    parser.add_argument("--split_type", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.3)
    args = parser.parse_args()
    main(args.stained_tiles_home, args.label_dir_path, args.divisionMethod, args.split_type, args.test_size)

