#-*-coding:utf-8-*-

from skimage import transform
from skimage import io
from keras.callbacks import ModelCheckpoint
from PIL import Image
from functools import partial

import os
import keras
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib
import skimage
import h5py
import cv2, shutil
import math


def build_model():
    base_path = '/media/DATA3/mdd/modified_polar_cell_data3/compare_data/128_128_h5_rotate12_flip0/'
    save_path = '/media/DATA3/mdd/modified_polar_cell_data3/compare_data/128_128_h5_rotate12_flip0_dualstream_firstcopy3/'

    train_h5_name = 'train_data_label.h5'
    train_data_path = base_path + train_h5_name
    f = h5py.File(train_data_path,'r')
    train_data = f['data'][:]
    train_label = f['label'][:]
    train_first_image_data = f['first_image'][:]
    f.close()

    repeat_train_first_image_data = np.repeat(train_first_image_data,3,1)
    permute_repeat_train_first_image_data = np.swapaxes(repeat_train_first_image_data, 1, 3)
    permute_repeat_train_first_image_data = np.swapaxes(permute_repeat_train_first_image_data, 1, 2)
    save_file_name = 'train_data_label.h5'

    f = h5py.File(save_path + save_file_name, 'w')
    f['data'] = train_data
    f['first_image'] = permute_repeat_train_first_image_data
    f['label'] = train_label
    f.close()

    train_h5_name = 'validate_data_label.h5'
    train_data_path = base_path + train_h5_name
    f = h5py.File(train_data_path,'r')
    train_data = f['data'][:]        
    train_label = f['label'][:]
    train_first_image_data = f['first_image'][:]
    f.close()

    repeat_train_first_image_data = np.repeat(train_first_image_data,3,1)

    permute_repeat_train_first_image_data = np.swapaxes(repeat_train_first_image_data, 1, 3)
    permute_repeat_train_first_image_data = np.swapaxes(permute_repeat_train_first_image_data, 1, 2)

    save_file_name = 'validate_data_label.h5'

    f = h5py.File(save_path + save_file_name, 'w')
    f['data'] = train_data
    f['first_image'] = permute_repeat_train_first_image_data
    f['label'] = train_label
    f.close()

    train_h5_name = 'test_data_label.h5'
    train_data_path = base_path + train_h5_name
    f = h5py.File(train_data_path,'r')
    train_data = f['data'][:]           
    train_label = f['label'][:]
    train_first_image_data = f['first_image'][:]
    f.close()

    repeat_train_first_image_data = np.repeat(train_first_image_data,3,1)

    permute_repeat_train_first_image_data = np.swapaxes(repeat_train_first_image_data, 1, 3)
    permute_repeat_train_first_image_data = np.swapaxes(permute_repeat_train_first_image_data, 1, 2)

    save_file_name = 'test_data_label.h5'

    f = h5py.File(save_path + save_file_name, 'w')
    f['data'] = train_data
    f['first_image'] = permute_repeat_train_first_image_data
    f['label'] = train_label
    f.close()

    test = 1
    return test

def self_main():

    build_model()


    test_end = 1


if __name__ == "__main__":

    self_main()




