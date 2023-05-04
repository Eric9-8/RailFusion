# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/17 17:51
import pickle
import os
import csv
import numpy as np
import random
from PIL import Image


def read_img(path):
    rail_path_label = []
    with open(os.path.join(path, 'images.csv')) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            rail_path_label.append((img, int(label)))

    return rail_path_label


def rail_dic(path_label):
    Rail = {}
    for i in range(len(path_label)):
        image_path = path_label[i][0]
        image = Image.open(image_path).convert('RGB')
        Rail.setdefault('Rail_img', {})[i] = [image]
        Rail.setdefault('Rail_label', {})[i] = path_label[i][1]

    return Rail


def gaf_dic(dic, path_label, idx):
    for i in range(len(path_label)):
        image_path = path_label[i][0]
        image = Image.open(image_path).convert('RGB')
        dic.setdefault('GAF_img_{j}'.format(j=idx), {})[i] = [image]

    return dic


def get_data(dic, key_s):
    return [(dic[key]) for key in key_s]


def label_process(labels):
    return [[x] for x in labels]


def dataset(rail, gaf):
    keys = list(rail['Rail_img'].keys())
    random.shuffle(keys)

    train_keys = keys[:int(0.8 * len(keys))]
    valid_keys = keys[int(0.8 * len(keys)):int(0.9 * len(keys))]
    test_keys = keys[int(0.9 * len(keys)):]

    RAIL_Train = get_data(RAIL['Rail_img'], train_keys)
    RAIL_Valid = get_data(RAIL['Rail_img'], valid_keys)
    RAIL_Test = get_data(RAIL['Rail_img'], test_keys)

    LABEL_Train = label_process(get_data(RAIL['Rail_label'], train_keys))
    LABEL_Valid = label_process(get_data(RAIL['Rail_label'], valid_keys))
    LABEL_Test = label_process(get_data(RAIL['Rail_label'], test_keys))

    Train = {'RAIL': RAIL_Train, 'LABEL': LABEL_Train}
    Valid = {'RAIL': RAIL_Valid, 'LABEL': LABEL_Valid}
    Test = {'RAIL': RAIL_Test, 'LABEL': LABEL_Valid}

    for idx, itm in enumerate(list(gaf.keys())):
        for tk in train_keys:
            train_gaf = gaf[itm][tk]
            Train.setdefault('GAF_{i}'.format(i=idx), []).append(train_gaf)
        for vk in valid_keys:
            valid_gaf = gaf[itm][vk]
            Valid.setdefault('GAF_{i}'.format(i=idx), []).append(valid_gaf)
        for tek in test_keys:
            test_gaf = gaf[itm][tek]
            Test.setdefault('GAF_{i}'.format(i=idx), []).append(test_gaf)

    Rail_datas = {'Train': Train, 'Valid': Valid, 'Test': Test}

    return Rail_datas


if __name__ == "__main__":

    path1 = r'D:\Pytorch\Fusion_transformer\data\Sample366\figures366_record'
    path2 = r'D:\Pytorch\Fusion_transformer\data\Sample366\GAFs_10_times_divide'

    root_files = os.listdir(path2)
    routes = [folder for folder in root_files]
    GAF = {}
    for index, item in enumerate(routes):
        route_dir = os.path.join(path2, item)
        gaf_img_label = read_img(route_dir)
        GAF = gaf_dic(GAF, gaf_img_label, index)
    rail_img_label = read_img(path1)
    RAIL = rail_dic(rail_img_label)

    Rail_dataset = dataset(RAIL, GAF)
    file = open('/Data/Sample366/Dataset_11_MODE.pkl', 'wb')
    pickle.dump(Rail_dataset, file)
    file.close()
