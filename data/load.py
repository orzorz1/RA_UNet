import numpy as np
import matplotlib
from matplotlib import pylab as plt
import nibabel as nib
import torch
from torch.utils.data import Dataset
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

#设置HU值（肝窗为-100到200）
def setHU(img_arr, min, max):
    width, height, deep = img_arr.shape
    for i in range(0, width):
        for j in range(0, height):
            for k in range(0,deep):
                if img_arr[i,j,k] <= min:
                    img_arr[i,j,k] = 0
                elif img_arr[i,j,k] >= max:
                    img_arr[i,j,k] = 1
                else:
                    img_arr[i,j,k] = (img_arr[i,j,k]-min)/(max-min)
    return img_arr

# 第一阶段的数据集处理
def set_label_1(img_arr):
    width, height, deep = img_arr.shape
    for i in range(0, width):
        for j in range(0, height):
            for k in range(0,deep):
                if img_arr[i,j,k] != 0:
                    img_arr[i,j,k] = 1
    return img_arr

# 加载数据集(一张)并设置HU值
def read_dataset(path):
    img = nib.load(path)
    img_arr = np.array(img.dataobj)
    img_arr = setHU(img_arr, -100, 200)
    return img_arr

# (width, height, deep) -> (deep, channel, width, height)
def reshape(pic):
    pic = np.transpose(pic, (2, 0, 1))
    pic = np.expand_dims(pic, axis=1)
    return pic

# 加载第一阶段的数据集
def read_label_1(path):
    img = nib.load(path)
    img_arr = np.array(img.dataobj)
    img_arr = set_label_1(img_arr)
    return img_arr

class dataset_step_1(Dataset):
    def __init__(self, index):
        path_x = "../dataset/train/volume-{i}.nii".format(i=index)
        x = read_dataset(path_x)
        x = reshape(x)
        path_y = "../dataset/train/segmentation-{i}.nii".format(i=index)
        y = read_label_1(path_y)
        print("数据加载完成，shape：",x.shape)
        y = reshape(y)
        imgs = []
        for i in range(x.shape[0]):
            imgs.append((x[i],y[i]))
        self.imgs = imgs

    def __getitem__(self, index):
        img, label = self.imgs[index]
        target = np.array(label)
        return torch.from_numpy(img).to(torch.float32), torch.from_numpy(target).long()


    def __len__(self):
        return len(self.imgs)

# path = "../dataset/train/volume-{index}.nii".format(index = 0)
# pic = read_dataset(path)
# print(pic.shape)
