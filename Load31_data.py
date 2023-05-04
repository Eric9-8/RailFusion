# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/17 14:07
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class RAIL_Data(Dataset):

    def __init__(self, preload_dic, config):
        self.seq_len = config.seq_len
        self.gaf_scale = config.gaf_scale
        self.rail_scale = config.rail_scale
        self.input_resolution = config.input_resolution

        self.rail = []
        self.gaf0 = []
        self.gaf1 = []
        self.gaf2 = []
        self.gaf3 = []
        self.gaf4 = []
        self.gaf5 = []
        self.gaf6 = []
        self.gaf7 = []
        self.gaf8 = []
        self.gaf9 = []
        self.gaf10 = []
        self.gaf11 = []
        self.gaf12 = []
        self.gaf13 = []
        self.gaf14 = []
        self.gaf15 = []
        self.gaf16 = []
        self.gaf17 = []
        self.gaf18 = []
        self.gaf19 = []
        self.gaf20 = []
        self.gaf21 = []
        self.gaf22 = []
        self.gaf23 = []
        self.gaf24 = []
        self.gaf25 = []
        self.gaf26 = []
        self.gaf27 = []
        self.gaf28 = []
        self.gaf29 = []
        self.label = []

        self.rail += preload_dic['RAIL']
        self.gaf0 += preload_dic['GAF_0']

        self.gaf1 += preload_dic['GAF_1']
        self.gaf2 += preload_dic['GAF_2']
        self.gaf3 += preload_dic['GAF_3']
        self.gaf4 += preload_dic['GAF_4']
        self.gaf5 += preload_dic['GAF_5']
        self.gaf6 += preload_dic['GAF_6']
        self.gaf7 += preload_dic['GAF_7']
        self.gaf8 += preload_dic['GAF_8']
        self.gaf9 += preload_dic['GAF_9']
        self.gaf10 += preload_dic['GAF_10']
        self.gaf11 += preload_dic['GAF_11']
        self.gaf12 += preload_dic['GAF_12']
        self.gaf13 += preload_dic['GAF_13']
        self.gaf14 += preload_dic['GAF_14']
        self.gaf15 += preload_dic['GAF_15']
        self.gaf16 += preload_dic['GAF_16']
        self.gaf17 += preload_dic['GAF_17']
        self.gaf18 += preload_dic['GAF_18']
        self.gaf19 += preload_dic['GAF_19']
        self.gaf20 += preload_dic['GAF_20']
        self.gaf21 += preload_dic['GAF_21']
        self.gaf22 += preload_dic['GAF_22']
        self.gaf23 += preload_dic['GAF_23']
        self.gaf24 += preload_dic['GAF_24']
        self.gaf25 += preload_dic['GAF_25']
        self.gaf26 += preload_dic['GAF_26']
        self.gaf27 += preload_dic['GAF_27']
        self.gaf28 += preload_dic['GAF_28']
        self.gaf29 += preload_dic['GAF_29']

        self.label += preload_dic['LABEL']

    def __len__(self):
        return len(self.rail)

    def __getitem__(self, index):
        data = dict()
        data['Rails'] = []
        data['GAF0s'] = []
        data['GAF1s'] = []
        data['GAF2s'] = []
        data['GAF3s'] = []
        data['GAF4s'] = []
        data['GAF5s'] = []
        data['GAF6s'] = []
        data['GAF7s'] = []
        data['GAF8s'] = []
        data['GAF9s'] = []
        data['GAF10s'] = []
        data['GAF11s'] = []
        data['GAF12s'] = []
        data['GAF13s'] = []
        data['GAF14s'] = []
        data['GAF15s'] = []
        data['GAF16s'] = []
        data['GAF17s'] = []
        data['GAF18s'] = []
        data['GAF19s'] = []
        data['GAF20s'] = []
        data['GAF21s'] = []
        data['GAF22s'] = []
        data['GAF23s'] = []
        data['GAF24s'] = []
        data['GAF25s'] = []
        data['GAF26s'] = []
        data['GAF27s'] = []
        data['GAF28s'] = []
        data['GAF29s'] = []

        data['Label'] = self.label[index]
        seq_rail = self.rail[index]
        seq_gaf0 = self.gaf0[index]

        seq_gaf1 = self.gaf1[index]
        seq_gaf2 = self.gaf2[index]
        seq_gaf3 = self.gaf3[index]
        seq_gaf4 = self.gaf4[index]
        seq_gaf5 = self.gaf5[index]
        seq_gaf6 = self.gaf6[index]
        seq_gaf7 = self.gaf7[index]
        seq_gaf8 = self.gaf8[index]
        seq_gaf9 = self.gaf9[index]
        seq_gaf10 = self.gaf10[index]
        seq_gaf11 = self.gaf11[index]
        seq_gaf12 = self.gaf12[index]
        seq_gaf13 = self.gaf13[index]
        seq_gaf14 = self.gaf14[index]
        seq_gaf15 = self.gaf15[index]
        seq_gaf16 = self.gaf16[index]
        seq_gaf17 = self.gaf17[index]
        seq_gaf18 = self.gaf18[index]
        seq_gaf19 = self.gaf19[index]
        seq_gaf20 = self.gaf20[index]
        seq_gaf21 = self.gaf21[index]
        seq_gaf22 = self.gaf22[index]
        seq_gaf23 = self.gaf23[index]
        seq_gaf24 = self.gaf24[index]
        seq_gaf25 = self.gaf25[index]
        seq_gaf26 = self.gaf26[index]
        seq_gaf27 = self.gaf27[index]
        seq_gaf28 = self.gaf28[index]
        seq_gaf29 = self.gaf29[index]

        for i in range(self.seq_len):
            data['Rails'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_rail[i], scale=self.rail_scale, crop=self.input_resolution)
            )))
            data['GAF0s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf0[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF1s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf1[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF2s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf2[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF3s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf3[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF4s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf4[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF5s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf5[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF6s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf6[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF7s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf7[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF8s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf8[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF9s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf9[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF10s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf10[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF11s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf11[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF12s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf12[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF13s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf13[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF14s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf14[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF15s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf15[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF16s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf16[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF17s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf17[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF18s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf18[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF19s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf19[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF20s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf20[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF21s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf21[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF22s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf22[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF23s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf23[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF24s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf24[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF25s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf25[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF26s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf26[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF27s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf27[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF28s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf28[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))
            data['GAF29s'].append(torch.from_numpy(np.array(
                scale_and_crop_image(seq_gaf29[i], scale=self.gaf_scale, crop=self.input_resolution)
            )))

            return data


Image_trans = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.5)
])


def scale_and_crop_image(image, scale=1, crop=128, trans=Image_trans):
    """
    Scale and crop a PIL image, returning a channels-first numpy array.
    """
    # image = Image.open(filename)
    image = trans(image)
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    image = np.asarray(im_resized)
    start_x = height // 2 - crop // 2
    start_y = width // 2 - crop // 2
    cropped_image = image[start_x:start_x + crop, start_y:start_y + crop]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image
