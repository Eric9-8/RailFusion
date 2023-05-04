# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/12 10:05
import pickle
import os
import csv
import numpy as np
import random
import cv2.cv2 as cv2
from PIL import Image
import torch


def read_img(path):
    rail_path_label = []
    with open(os.path.join(path, 'images.csv')) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            rail_path_label.append((img, int(label)))
    return rail_path_label


def mk_dic(path_label, mode='Rail'):
    Rail = {}
    GAF = {}
    GAF_img = []
    GAF_label = []
    count = 0
    if mode == 'Rail':
        for i in range(len(path_label)):
            image_path = path_label[i][0]
            image = Image.open(image_path).convert('RGB')
            Rail.setdefault('Rail_img', {})[i] = image
            Rail.setdefault('Rail_label', {})[i] = path_label[i][1]
    else:
        for j in range(len(path_label)):
            image_path = path_label[j][0]
            image = Image.open(image_path).convert('RGB')
            GAF_img.append(image)
            GAF_label.append(path_label[j][1])

        imgs = [GAF_img[n:n + 10] for n in range(0, len(GAF_img), 10)]
        labels = np.array(GAF_label).reshape((140, 10))

        for index, item in enumerate(imgs):
            GAF.setdefault('GAF_img', {})[count] = item
            GAF.setdefault('GAF_label', {})[count] = labels[index][0]
            GAF.setdefault('GAF10_label', {})[count] = labels[index]
            count += 1
    return Rail, GAF


def get_data(dic, key_s):
    return [(dic[key]) for key in key_s]
    # return np.array([(dic[key]) for key in key_s])


def label_process(labels):
    # return [[x] for x in labels]
    return np.array([[x] for x in labels])


if __name__ == "__main__":
    # rail_img = 'D:\\Pytorch\\Fusion_lowrank\\data\\figures366_record'
    # gaf_img = 'D:\\Pytorch\\Fusion_transformer\\data\\Sample366\\GAFs'

    # rail_img = 'D:\\Pytorch\\Fusion_lowrank\\data\\figures'
    # gaf_img = 'D:\\Pytorch\\Fusion_transformer\\data\\Sample140\\GAFs'

    # rail_img = 'D:\\Pytorch\\Fusion_lowrank\\data\\figures366_record'
    # gaf_img = 'D:\\Pytorch\\Fusion_transformer\\data\\Sample366\\GAFs_10_times'

    rail_img = 'D:\\Pytorch\\Fusion_lowrank\\data\\figures'
    gaf_img = 'D:\\Pytorch\\Fusion_transformer\\data\\Sample140\\GAFs_10_times'

    Rail_path_label = read_img(rail_img)
    GAF_path_label = read_img(gaf_img)

    RAIL, _ = mk_dic(Rail_path_label, mode='Rail')
    _, GAF = mk_dic(GAF_path_label, mode='GAF')

    keys = list(RAIL['Rail_img'].keys())
    random.shuffle(keys)

    train_keys = keys[:int(0.7 * len(keys))]
    valid_keys = keys[int(0.7 * len(keys)):int(0.9 * len(keys))]
    test_keys = keys[int(0.9 * len(keys)):]

    RAIL_Train = get_data(RAIL['Rail_img'], train_keys)
    RAIL_Valid = get_data(RAIL['Rail_img'], valid_keys)
    RAIL_Test = get_data(RAIL['Rail_img'], test_keys)

    GAF_Train = get_data(GAF['GAF_img'], train_keys)
    GAF_Valid = get_data(GAF['GAF_img'], valid_keys)
    GAF_Test = get_data(GAF['GAF_img'], test_keys)

    LABEL_Train = label_process(get_data(RAIL['Rail_label'], train_keys))
    LABEL_Valid = label_process(get_data(RAIL['Rail_label'], valid_keys))
    LABEL_Test = label_process(get_data(RAIL['Rail_label'], test_keys))

    LABEL10_Train = label_process(get_data(GAF['GAF10_label'], train_keys))
    LABEL10_Valid = label_process(get_data(GAF['GAF10_label'], valid_keys))
    LABEL10_Test = label_process(get_data(GAF['GAF10_label'], test_keys))

    Train = {'RAIL': RAIL_Train, 'GAF': GAF_Train, 'LABEL': LABEL_Train, 'LABEL10': LABEL10_Train}
    Valid = {'RAIL': RAIL_Valid, 'GAF': GAF_Valid, 'LABEL': LABEL_Valid, 'LABEL10': LABEL10_Valid}
    Test = {'RAIL': RAIL_Test, 'GAF': GAF_Test, 'LABEL': LABEL_Valid, 'LABEL10': LABEL10_Test}

    Rail_dataset = {'Train': Train, 'Valid': Valid, 'Test': Test}
    file = open('D:\\Pytorch\\Fusion_transformer\\data\\Sample140\\Rail_GAF10_PIL.pkl', 'wb')
    pickle.dump(Rail_dataset, file)
    file.close()
