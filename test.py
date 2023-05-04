# 上海工程技术大学
# 崔嘉亮
# 开发时间：2022/6/11 9:19
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
from pyts.datasets import load_gunpoint
import pickle
import torch

# X, _, _, _ = load_gunpoint(return_X_y=True)
#
# gaf = GramianAngularField()
# X_gaf = gaf.fit_transform(X)
#
# fig = plt.figure(figsize=(10, 5))

# Rail_GAF_data = pickle.load(open(r"D:\Pytorch\Fusion_transformer\data\Sample366\Dataset_11_MODE.pkl", 'rb'))
# Train = Rail_GAF_data['Train']
# target0 = Train['LABEL']
# target1 = torch.stack(Train['LABEL'], dim=1)
# # cc = []
# # cc += Train.item()['RAIL']
# for i in range(1):
#     print(i)

print('========================')
