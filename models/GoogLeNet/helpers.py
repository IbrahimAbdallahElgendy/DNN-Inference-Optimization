# coding=utf-8
"""
Helper functions for GoogLeNet model
Adapted from: https://github.com/km1414/CNN-models
* Googlenet.py *的辅助函数。 需要访问互联网才能下载训练数据集。
"""

import numpy as np
import pylab as pl
import pickle
import os
import download
from keras.utils.np_utils import to_categorical


data_path = "data/CIFAR-10/"
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
img_size = 32
num_channels = 3
img_size_flat = img_size * img_size * num_channels
num_classes = 10
_num_files_train = 5
_images_per_file = 10000
_num_images_train = _num_files_train * _images_per_file

# 取得数据存放地址
def _get_file_path(filename=""):

    return os.path.join(data_path, "cifar-10-batches-py/", filename)

# 加载pickle文件
def _unpickle(filename):

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:

        # data = pickle.load(file)
        data = pickle.load(file, encoding='bytes')

    return data

# 转换images
def _convert_images(raw):

    # 将原始图像从data-files转换为floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # 重新排列数组的索引.
    images = images.transpose([0, 2, 3, 1])

    return images

# 加载数据
def _load_data(filename):

    # 加载pickled data-file
    data = _unpickle(filename)

    # 获得原生图片raw images
    raw_images = data[b'data']

    # 获取每个图像的class-numbers.转换为numpy-array。
    cls = np.array(data[b'labels'])

    # 转换images.
    images = _convert_images(raw_images)

    return images, cls

# 下载并解压数据
def download_and_extract_data():

    download.download(url=data_url, path=data_path, kind='tar.gz', progressbar=True, replace=False, verbose=True)

# 加载类名
def load_class_names():

    # 从pickled文件中加载类名
    raw = _unpickle(filename="batches.meta")[b'label_names']

    # 从二进制字符串转换.
    names = [x.decode('utf-8') for x in raw]

    return names

# 加载训练数据
def load_training_data():

    # 为images和class-numbers预先分配arrays,以提高效率.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # 当前batch的Begin-index
    begin = 0

    # 对每一个data-file.
    for i in range(_num_files_train):
        # 从data-file中加载images和class-numbers
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        # 这个batch中images的编号
        num_images = len(images_batch)

        # 当前batch的End-index
        end = begin + num_images

        # 存储images到array中
        images[begin:end, :] = images_batch

        # 存储class-numbers到array中
        cls[begin:end] = cls_batch

        # 下一个batch的begin-index是当前end-index
        begin = end

    return images, cls, to_categorical(cls, num_classes=num_classes)

# 加载测试数据
def load_test_data():

    images, cls = _load_data(filename="test_batch")

    return images, cls, to_categorical(cls, num_classes=num_classes)


# 绘图结果
def plot_results(results):
    pl.figure()

    pl.subplot(121)
    pl.plot(results['train_acc'])
    pl.title('Accuracy:')
    pl.plot(results['test_acc'])
    pl.legend(('Train', 'Test'))

    pl.subplot(122)
    pl.plot(results['train_loss'])
    pl.title('Cost:')
    pl.plot(results['test_loss'])
    pl.legend(('Train', 'Test'))
