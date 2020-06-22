# -*- coding: utf-8 -*-
"""
    Created on Wednesday, Jun 10 2020

    Author          ：Yu Du
    Email           : yuduseu@gmail.com
    Last edit date  : Tuesday, Jun 23 2020

South East University Automation College, 211189 Nanjing China
"""

from torch.utils import data
import torch
import os
import cv2


class CustomDataset(data.Dataset):
    """
    Load custom dataset by giving the data folder path
    """
    def __init__(self, path, train=True, img_size=28, in_memory=True, max_amount=100000):
        """
        :param path: (string) The directory for the custom dataset
        :param img_size: (int) the shape of the returned image data,
                            which is in square and if the original image is not square, it will cropped in the middle
        :param in_memory: (bool) if True: load the whole dataset into memory,
                                            which would be faster in training but occupy lots of memory
                                 else: load the data from disk when calling __getitem__(),
                                            which would be occupy little memory but slower
        """
        super(CustomDataset, self).__init__()
        self.img_size = img_size
        self.in_memory = in_memory
        self.img_names = []
        self.train = train
        for root, _, files in os.walk(path):
            train_num = 2*len(files) // 3
            if self.train:
                low = 0
                high = train_num
            else:
                low = train_num + 1
                high = len(files)
            for i in range(low, high):
                self.img_names.append(os.path.join(root, files[i]))
        self.num = len(self.img_names)
        self.imgs = []
        if self.in_memory:
            # if in_memory, read the image file in advance, 220k images may cost about 2.2G memory
            for i, name in enumerate(self.img_names):
                img = cv2.imread(name)
                resized = self.__pre_process(img)
                self.imgs.append(resized)
                if i % 1000 == 0:
                    print('{0} images has been load into memory'.format(i))

    def __getitem__(self, idx):
        """
        return: (tensor) data send to network
        obj[idx] == obj.__getitem__(idx)
        """
        label = -1
        if self.in_memory:
            return self.imgs[idx], label
        else:
            img = cv2.imread(self.img_names[idx])
            return self.__pre_process(img), label

    def __len__(self):
        return self.num

    def __pre_process(self, img):
        """
        :param img: (ndarray) image read from file by opencv
        :return: (tensor) data send to network
        """
        h, w, _ = img.shape
        # This way may contain more background that affect the learning of the model
        # size = min(h, w)
        
        # contain less background make the dataset cleaner
        size = 128
        
        # crop around the center point
        cropped = img[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size, :]
        # the resized size should correspond with the network
        resized = cv2.resize(cropped, (self.img_size, self.img_size))
        # image read by opencv is BGR, convert it to common RGB
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # channel of image read by opencv is HWC
        # while for the network, the last C(channel, RGB) is treated as the number of feature
        # so it should be changed to CHW
        resized = resized.swapaxes(1, 2).swapaxes(0, 1)
        # normalization
        resized = torch.from_numpy(resized).float() / 255
        return resized

    def get_img(self, idx):
        """
        :param idx: (int) index
        :return: (ndarray) cropped and resized BGR image
        """
        img = cv2.imread(self.img_names[idx])
        h, w, _ = img.shape
        size = 128
        cropped = img[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size, :]
        resized = cv2.resize(cropped, (self.img_size, self.img_size))
        return resized


if __name__ == '__main__':
    # Here is to show the cropped image, to see what data will be send to the network
    dataset_path = '../../data/img_align_celeba'
    img_size = 32
    dataset = CustomDataset(
        dataset_path,
        img_size=img_size,
        in_memory=False
    )
    cv2.imshow('Cropped', dataset.get_img(0))
    cv2.waitKey(0)
