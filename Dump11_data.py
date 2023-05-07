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
    """
    读取图片路径和标签
    """
    rail_path_label = []
    with open(os.path.join(path, 'images.csv')) as f:
        reader = csv.reader(f)
        for row in reader:
            img, label = row
            rail_path_label.append((img, int(label)))

    return rail_path_label


def rail_dic(path_label):
    """
    读取钢轨图像数据和标签
    """
    Rail = {}
    for i in range(len(path_label)):
        image_path = path_label[i][0]
        image = Image.open(image_path).convert('RGB')
        Rail.setdefault('Rail_img', {})[i] = [image]
        Rail.setdefault('Rail_label', {})[i] = path_label[i][1]

    return Rail


# 多模态
def gaf_dic(dic, path_label, idx):
    """
    滑动窗口下读取GAF图像数据和标签
    """
    for i in range(len(path_label)):
        image_path = path_label[i][0]
        image = Image.open(image_path).convert('RGB')
        dic.setdefault('GAF_img_{j}'.format(j=idx), {})[i] = [image]

    return dic

# 单模态
# def gaf_dic(path_label):
#     """
#     非滑动窗口下读取GAF图像数据和标签
#     """
#     GAF = {}
#     for i in range(len(path_label)):
#         image_path = path_label[i][0]
#         image = Image.open(image_path).convert('RGB')
#         GAF.setdefault('GAF_img', {})[i] = [image]
#         # GAFs.setdefault('Rail_label', {})[i] = path_label[i][1]

    # return GAF


def get_data(dic, key_s):
    return [(dic[key]) for key in key_s]


def label_process(labels):
    return [[x] for x in labels]


def dataset(rail, gaf):
    """
    制作数据集，按照8/1/1划分，一张钢轨图像对应不同数量的GAF图像
    """
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
    Test = {'RAIL': RAIL_Test, 'LABEL': LABEL_Test}

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
    # gaf image
    path2 = r'D:\Pytorch\Fusion_transformer\data\Sample366\GAFs_30_times_divide'
    path4 = r'D:\Pytorch\Fusion_transformer\data\Sample140\GAFs_30_times_divide_fusion_1'
    path8 = r'D:\Pytorch\Fusion_transformer\data\Sample140\GAFs'

    path9 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_imf_channel4'
    path10 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_imf_fusion_5'

    path12 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_fusion_5'

    path13 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_rec_channel4'
    path14 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_rec_fusion_1'

    path15 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_rec_TQWT_channel4'
    path16 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_rec_TQWT_fusion_5'

    path17 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_rec_TQWT_fusion_nowindow_1'

    path18 = r'D:\Pytorch\Fusion_transformer\data\Sample75\GAFs_30_times_ceemdan_fusion_1'

    # rail image
    path1 = r'D:\Pytorch\Fusion_transformer\data\Sample366\figures366_record'
    path3 = r'D:\Pytorch\Fusion_lowrank\data\original_figures'
    path5 = r'D:\Pytorch\Fusion_transformer\data\Sample140\Rail_image_cutting'
    path6 = r'D:\Pytorch\Fusion_transformer\data\Sample140\Rail_square_image'

    path11 = r'D:\Pytorch\Fusion_transformer\data\Sample75\RailOriginal'

    # gray image
    path7 = r'D:\Pytorch\Fusion_transformer\data\Sample140\Grays_fusion'

    # 多模态
    root_files = os.listdir(path18)
    routes = [folder for folder in root_files]
    GAF = {}
    for index, item in enumerate(routes):
        route_dir = os.path.join(path18, item)
        gaf_img_label = read_img(route_dir)
        GAF = gaf_dic(GAF, gaf_img_label, index)

    # 单模态
    # gaf_img_label = read_img(path17)
    # GAF = gaf_dic(gaf_img_label)

    rail_img_label = read_img(path11)
    RAIL = rail_dic(rail_img_label)

    Rail_dataset = dataset(RAIL, GAF)
    file = open('D:/Pytorch/Fusion_transformer/data/Sample75/Dataset_30_ceemdan_fusion_1.pkl', 'wb')
    pickle.dump(Rail_dataset, file)
    file.close()
