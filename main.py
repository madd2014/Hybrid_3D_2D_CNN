#-*-coding:utf-8-*-

from skimage import transform
from skimage import io
# from fuse_resnet50 import unet_model_3d
# from fuse_vgg16 import unet_model_3d
# from fuse_vgg19 import unet_model_3d
# from fuse_inceptionV3 import unet_model_3d
# from fuse_inceptionResNetV2 import unet_model_3d
# from fuse_densenet121 import unet_model_3d
# from fuse_densenet201 import unet_model_3d
# from fuse_xception import unet_model_3d
from fuse_mobilenetV2 import unet_model_3d

from keras.callbacks import ModelCheckpoint
from PIL import Image
from functools import partial
import scipy.io as scio

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

from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

bit_max_value = 4095
polar_img_num = 30
resize_width = 128
resize_height = 128
aug_rotate_num = 12
aug_flip_num = 0        

aug_scale_num = 0         
aug_scale_step = 0.2

aug_translation_num = 0      
aug_translation_step = 5     

aug_blur_num = 0
aug_blur_step = 1        
blur_wsize = (5,5)

aug_gnoise_num = 0     
aug_gnoise_step = 0.002  

mat_data_inner_name = 'cat_csv_data'

first_input_shape = (1,polar_img_num-1, resize_width, resize_height)
second_input_shape = (1,resize_width, resize_height)

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
config = dict()
config["pool_size"] = (2, 2)               
config["image_shape"] = (polar_img_num-1, resize_width, resize_height)         
config["all_modalities"] = ["polar"]
config["training_modalities"] = config["all_modalities"]        
config["nb_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True                        
config["n_epochs"] = 500                        
config["patience"] = 10                         
config["early_stop"] = 50                        
config["initial_learning_rate"] = 0.1
config["learning_rate_drop"] = 0.5                
config["validation_split"] = 0.8      
config["overwrite"] = False               
num_classes = 4

def orig_save_mat_into_h5(base_path, mat_data_filename, h5_data_filename, class_names):
    # train
    train_data_path = base_path + mat_data_filename + 'train/'
    train_data_lists = os.listdir(train_data_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_data_path + each_mat_name
        train_mat_data_info = sio.loadmat(mat_path)
        train_mat_data = train_mat_data_info[mat_data_inner_name]

        first_resized_polar_image = transform.resize(train_mat_data[0,:,:].astype('float64'), (resize_width, resize_height))
        first_resized_polar_image = first_resized_polar_image/bit_max_value

        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image,axis = 0)
        ext_first_resized_polar_image = np.expand_dims(ext_first_resized_polar_image,axis = 0)

        ext_first_polar_image = np.expand_dims(train_mat_data[0,:,:],axis = 0)         
        ext_first_polar_image = np.repeat(ext_first_polar_image,polar_img_num-1,axis = 0)       
        diff_train_mat_data = (train_mat_data[1:,:,:].astype('float64') - ext_first_polar_image.astype('float64'))/bit_max_value

        resize_train_mat_data = transform.resize(diff_train_mat_data, (polar_img_num-1, resize_width, resize_height))
        diff_train_mat_data = np.expand_dims(resize_train_mat_data, axis=0)
        diff_train_mat_data = np.expand_dims(diff_train_mat_data, axis=0)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])
        # one_hot_train_mat_label = keras.utils.to_categorical(train_mat_label, num_classes)

        if counter == 0:
            batch_diff_train_data = diff_train_mat_data
            batch_train_label = train_mat_label
            batch_first_resized_polar_image = ext_first_resized_polar_image
        else:
            batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
            batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, ext_first_resized_polar_image), axis=0)
        counter = counter + 1

    save_h5_path = base_path + h5_data_filename
    if not(os.path.exists(save_h5_path)):
        os.makedirs(save_h5_path)

    f = h5py.File(save_h5_path + 'train_data_label.h5','w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    # validate
    validate_data_path = base_path + mat_data_filename + 'validate/'
    validate_data_lists = os.listdir(validate_data_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_data_path + each_mat_name
        validate_mat_data_info = sio.loadmat(mat_path)
        validate_mat_data = validate_mat_data_info[mat_data_inner_name]

        first_resized_polar_image = transform.resize(validate_mat_data[0,:,:].astype('float64'), (resize_width, resize_height))
        first_resized_polar_image = first_resized_polar_image/bit_max_value
        # first_resized_polar_image = (first_resized_polar_image-np.mean(first_resized_polar_image))/np.std(first_resized_polar_image)
        # first_resized_polar_image = (first_resized_polar_image - first_resized_polar_image.min()) / (
        #             first_resized_polar_image.max() - first_resized_polar_image.min())
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image,axis = 0)
        ext_first_resized_polar_image = np.expand_dims(ext_first_resized_polar_image, axis=0)

        ext_first_polar_image = np.expand_dims(validate_mat_data[0,:,:],axis = 0)             
        ext_first_polar_image = np.repeat(ext_first_polar_image,polar_img_num-1,axis = 0)           
        diff_validate_mat_data = (validate_mat_data[1:,:,:].astype('float64') - ext_first_polar_image.astype('float64'))/bit_max_value
        # diff_validate_mat_data = (diff_validate_mat_data - np.mean(diff_validate_mat_data)) / np.std(diff_validate_mat_data)
        # diff_validate_mat_data = (diff_validate_mat_data - diff_validate_mat_data.min()) / (
        #             diff_validate_mat_data.max() - diff_validate_mat_data.min())

        resize_validate_mat_data = transform.resize(diff_validate_mat_data, (polar_img_num-1, resize_width, resize_height))
        diff_validate_mat_data = np.expand_dims(resize_validate_mat_data, axis=0)
        diff_validate_mat_data = np.expand_dims(diff_validate_mat_data, axis=0)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])
        # one_hot_validate_mat_label = keras.utils.to_categorical(validate_mat_label, num_classes)

        if counter == 0:
            batch_diff_validate_data = diff_validate_mat_data
            batch_validate_label = validate_mat_label
            batch_first_resized_polar_image = ext_first_resized_polar_image
        else:
            batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
            batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, ext_first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(save_h5_path + 'validate_data_label.h5','w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    # test
    test_data_path = base_path + mat_data_filename + 'test/'
    test_data_lists = os.listdir(test_data_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_data_path + each_mat_name
        test_mat_data_info = sio.loadmat(mat_path)
        test_mat_data = test_mat_data_info[mat_data_inner_name]

        first_resized_polar_image = transform.resize(test_mat_data[0,:,:].astype('float64'), (resize_width, resize_height))
        first_resized_polar_image = first_resized_polar_image / bit_max_value
        # first_resized_polar_image = (first_resized_polar_image-np.mean(first_resized_polar_image))/np.std(first_resized_polar_image)
        # first_resized_polar_image = (first_resized_polar_image - first_resized_polar_image.min()) / (
        #             first_resized_polar_image.max() - first_resized_polar_image.min())
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image,axis = 0)
        ext_first_resized_polar_image = np.expand_dims(ext_first_resized_polar_image, axis=0)

        ext_first_polar_image = np.expand_dims(test_mat_data[0,:,:],axis = 0)           
        ext_first_polar_image = np.repeat(ext_first_polar_image,polar_img_num-1,axis = 0)  
        diff_test_mat_data = (test_mat_data[1:,:,:].astype('float64') - ext_first_polar_image.astype('float64'))/bit_max_value
        # diff_test_mat_data = (diff_test_mat_data - np.mean(diff_test_mat_data)) / np.std(diff_test_mat_data)
        # diff_test_mat_data = (diff_test_mat_data - diff_test_mat_data.min()) / (
        #             diff_test_mat_data.max() - diff_test_mat_data.min())

        resize_test_mat_data = transform.resize(diff_test_mat_data, (polar_img_num-1, resize_width, resize_height))
        diff_test_mat_data = np.expand_dims(resize_test_mat_data, axis=0)
        diff_test_mat_data = np.expand_dims(diff_test_mat_data, axis=0)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])
        # one_hot_test_mat_label = keras.utils.to_categorical(test_mat_label, num_classes)

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = ext_first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, ext_first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(save_h5_path + 'test_data_label.h5','w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_end = 1


def my_get_train_validate_test_label(data_path,class_names):
    data_lists = os.listdir(data_path)
    train_label = []
    validate_label = []
    test_label = []
    for data_list in data_lists:
        data_file_path = data_path + data_list
        sample_data = os.listdir(data_file_path)
        for mat_data in sample_data:

            cls_name = mat_data[0:7]
            if cls_name == class_names[0]:
                sample_label = 0
            elif cls_name == class_names[1]:
                sample_label = 1
            elif cls_name == class_names[2]:
                sample_label = 2
            else:
                sample_label = 3
            if data_list == 'train':
                train_label.append(sample_label)
            elif data_list == 'test':
                test_label.append(sample_label)
            else:
                validate_label.append(sample_label)
    return train_label, validate_label, test_label
    test_end = 1

def my_get_train_generator_v2(train_data_path,train_batch_size):
    f = h5py.File(train_data_path,'r')
    train_data = f['data'][:]
    train_label = f['label'][:]
    train_first_image_data = f['first_image'][:]
    f.close()

    train_data_num = train_data.shape[0]
    num_train_batches = int(train_data_num/train_batch_size)

    while True:
        # permutation = list(np.random.permutation(validate_data_num))          
        permutation = list(range(train_data_num))                       
        for i in range(num_train_batches):
            permute_indexes = permutation[i*train_batch_size:(i+1)*train_batch_size]
            batch_train_data = train_data[permute_indexes,:,:,:,:]
            batch_train_label = train_label[permute_indexes,]
            yield [batch_train_data, batch_train_label], [batch_train_label,batch_train_label]  


def my_get_validate_generator_v2(validate_data_path,validate_batch_size):

    f = h5py.File(validate_data_path,'r')
    validate_data = f['data'][:]
    validate_label = f['label'][:]
    validate_first_image_data = f['first_image'][:]
    f.close()

    validate_data_num = validate_data.shape[0]
    num_validate_batches = int(validate_data_num/validate_batch_size)

    while True:
        # permutation = list(np.random.permutation(validate_data_num))       
        permutation = list(range(validate_data_num))                            
        for i in range(num_validate_batches):
            permute_indexes = permutation[i*validate_batch_size:(i+1)*validate_batch_size]
            batch_validate_data = validate_data[permute_indexes,:,:,:,:]
            batch_validate_label = validate_label[permute_indexes,]
            yield [batch_validate_data, batch_validate_label], [batch_validate_label, batch_validate_label]  


def my_get_test_generator_v2(test_data_path,test_batch_size):

    f = h5py.File(test_data_path,'r')
    test_data = f['data'][:]
    test_label = f['label'][:]
    test_first_image_data = f['first_image'][:]
    f.close()

    test_data_num = test_data.shape[0]
    num_test_batches = int(test_data_num/test_batch_size)

    while True:
        # permutation = list(np.random.permutation(test_data_num))         
        permutation = list(range(test_data_num))                          
        for i in range(num_test_batches):
            permute_indexes = permutation[i*test_batch_size:(i+1)*test_batch_size]
            batch_test_data = test_data[permute_indexes,:,:,:,:]
            batch_test_label = test_label[permute_indexes,]
            yield batch_test_data, batch_test_label    


def my_get_training_validate_test_generators_v2(data_path,train_batch_size,validate_batch_size,test_batch_size):

    train_data_file = data_path + 'train_data_label.h5'
    validate_data_file = data_path + 'validate_data_label.h5'
    test_data_file = data_path + 'test_data_label.h5'

    train_generator = my_get_train_generator_v2(train_data_file,train_batch_size)
    validate_generator = my_get_validate_generator_v2(validate_data_file,validate_batch_size)
    test_generator = my_get_test_generator_v2(test_data_file,test_batch_size)

    return train_generator, validate_generator, test_generator

def get_second_train_generator(train_data_path,train_batch_size):  

    f = h5py.File(train_data_path,'r')
    train_data = f['data'][:]
    train_label = f['label'][:]
    train_first_image_data = f['first_image'][:]
    f.close()

    train_data_num = train_first_image_data.shape[0]
    num_train_batches = int(train_data_num/train_batch_size)

    while True:
        # permutation = list(np.random.permutation(validate_data_num))   
        permutation = list(range(train_data_num))                           
        for i in range(num_train_batches):
            permute_indexes = permutation[i*train_batch_size:(i+1)*train_batch_size]
            batch_train_data = train_data[permute_indexes, :, :, :, :]
            batch_train_first_image_data = train_first_image_data[permute_indexes,:,:,:]
            batch_train_label = train_label[permute_indexes,]
            yield [batch_train_data, batch_train_first_image_data, batch_train_label], [batch_train_label,batch_train_label] 


def get_second_validate_generator(validate_data_path,validate_batch_size):

    f = h5py.File(validate_data_path,'r')
    validate_data = f['data'][:]
    validate_label = f['label'][:]
    validate_first_image_data = f['first_image'][:]
    f.close()

    validate_data_num = validate_first_image_data.shape[0]
    num_validate_batches = int(validate_data_num/validate_batch_size)

    while True:
        # permutation = list(np.random.permutation(validate_data_num))     
        permutation = list(range(validate_data_num))                     
        for i in range(num_validate_batches):
            permute_indexes = permutation[i*validate_batch_size:(i+1)*validate_batch_size]
            batch_validate_data = validate_data[permute_indexes, :, :, :, :]
            batch_validate_first_image_data = validate_first_image_data[permute_indexes,:,:,:]
            batch_validate_label = validate_label[permute_indexes,]
            yield [batch_validate_data, batch_validate_first_image_data, batch_validate_label], [batch_validate_label,batch_validate_label]


def get_second_test_generator(test_data_path,test_batch_size):

    f = h5py.File(test_data_path,'r')
    test_data = f['data'][:]
    test_label = f['label'][:]
    test_first_image_data = f['first_image'][:]
    f.close()

    test_data_num = test_first_image_data.shape[0]
    num_test_batches = int(test_data_num/test_batch_size)

    while True:
        # permutation = list(np.random.permutation(test_data_num)) 
        permutation = list(range(test_data_num))                  
        for i in range(num_test_batches):
            permute_indexes = permutation[i*test_batch_size:(i+1)*test_batch_size]
            batch_test_data = test_data[permute_indexes, :, :, :, :]
            batch_test_first_image_data = test_first_image_data[permute_indexes,:,:,:]
            batch_test_label = test_label[permute_indexes,]
            yield [batch_test_data, batch_test_first_image_data], batch_test_label 


def get_second_training_validate_test_generators(data_path,train_batch_size,validate_batch_size,test_batch_size):

    train_data_file = data_path + 'train_data_label.h5'
    validate_data_file = data_path + 'validate_data_label.h5'
    test_data_file = data_path + 'test_data_label.h5'

    train_generator = get_second_train_generator(train_data_file,train_batch_size)
    validate_generator = get_second_validate_generator(validate_data_file,validate_batch_size)
    test_generator = get_second_test_generator(test_data_file,test_batch_size)

    return train_generator, validate_generator, test_generator


def flip_rotate_aug_and_save_to_h5(base_path, mat_data_filename, class_names):

    aug_mat_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_rotate' + str(aug_rotate_num) + '_flip' + str(aug_flip_num) + '/'
    if not(os.path.exists(aug_mat_path)):
        os.makedirs(aug_mat_path)

    train_mat_path = base_path + mat_data_filename + '/train/'
    train_data_lists = os.listdir(train_mat_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_mat_path + each_mat_name

        train_mat_data_info = sio.loadmat(mat_path)
        train_mat_data = train_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_train_mat_data = transform.resize(train_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_train_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])
        # one_hot_train_mat_label = keras.utils.to_categorical(train_mat_label, num_classes)

        for angle_num in range(aug_rotate_num):
            angle_step = 360/aug_rotate_num
            this_rotate_angle = angle_num*angle_step
            M = cv2.getRotationMatrix2D(((resize_width - 1) / 2, (resize_height - 1) / 2), this_rotate_angle, 1)
            rotated_mat_data = cv2.warpAffine(swap_resize_mat_data2, M, (resize_width, resize_height), borderMode=1)

            restore_mat_data1 = np.swapaxes(rotated_mat_data, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            train_mat_data = np.expand_dims(train_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(train_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_train_mat_data = train_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = train_mat_data[:, :, 0, :, :]


            if counter == 0:
                batch_diff_train_data = diff_train_mat_data
                batch_train_label = train_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1

            else:
                batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
                batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,0)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0)    # 扩展
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0)    # 扩展
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,-1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0)    # 扩展
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'train_data_label.h5', 'w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    validate_mat_path = base_path + mat_data_filename + '/validate/'
    validate_data_lists = os.listdir(validate_mat_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_mat_path + each_mat_name

        validate_mat_data_info = sio.loadmat(mat_path)
        validate_mat_data = validate_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_validate_mat_data = transform.resize(validate_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_validate_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])
        # one_hot_validate_mat_label = keras.utils.to_categorical(validate_mat_label, num_classes)

        for angle_num in range(aug_rotate_num):
            angle_step = 360/aug_rotate_num
            this_rotate_angle = angle_num*angle_step
            M = cv2.getRotationMatrix2D(((resize_width - 1) / 2, (resize_height - 1) / 2), this_rotate_angle, 1)
            rotated_mat_data = cv2.warpAffine(swap_resize_mat_data2, M, (resize_width, resize_height), borderMode=1)

            restore_mat_data1 = np.swapaxes(rotated_mat_data, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(validate_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_validate_mat_data = validate_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = validate_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_validate_data = diff_validate_mat_data
                batch_validate_label = validate_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
                batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,0)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)  
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0) 
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,-1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0) 
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'validate_data_label.h5', 'w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_mat_path = base_path + mat_data_filename + '/test/'
    test_data_lists = os.listdir(test_mat_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_mat_path + each_mat_name

        test_mat_data_info = sio.loadmat(mat_path)
        test_mat_data = test_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_test_mat_data = transform.resize(test_mat_data, (polar_img_num, resize_width, resize_height))

        test_mat_data = np.expand_dims(resize_test_mat_data, axis=0)
        test_mat_data = np.expand_dims(test_mat_data, axis=0)

        ext_first_resized_polar_image = np.expand_dims(test_mat_data[:, :, 0, :, :], axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_test_mat_data = test_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])
        # one_hot_test_mat_label = keras.utils.to_categorical(test_mat_label, num_classes)

        first_resized_polar_image = test_mat_data[:, :, 0, :, :]

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(aug_mat_path + 'test_data_label.h5', 'w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_end = 1

def rotate_aug_and_save_to_h5(base_path, mat_data_filename, class_names):

    aug_mat_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + str(aug_rotate_num) + '/'
    if not(os.path.exists(aug_mat_path)):
        os.makedirs(aug_mat_path)

    train_mat_path = base_path + mat_data_filename + '/train/'
    train_data_lists = os.listdir(train_mat_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_mat_path + each_mat_name

        train_mat_data_info = sio.loadmat(mat_path)
        train_mat_data = train_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_train_mat_data = transform.resize(train_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_train_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])
        # one_hot_train_mat_label = keras.utils.to_categorical(train_mat_label, num_classes)

        for angle_num in range(aug_rotate_num):
            angle_step = 360/aug_rotate_num
            this_rotate_angle = angle_num*angle_step
            M = cv2.getRotationMatrix2D(((resize_width - 1) / 2, (resize_height - 1) / 2), this_rotate_angle, 1)
            rotated_mat_data = cv2.warpAffine(swap_resize_mat_data2, M, (resize_width, resize_height), borderMode=1)

            restore_mat_data1 = np.swapaxes(rotated_mat_data, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            train_mat_data = np.expand_dims(train_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(train_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_train_mat_data = train_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = train_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_train_data = diff_train_mat_data
                batch_train_label = train_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1

            else:
                batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
                batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'train_data_label.h5', 'w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    validate_mat_path = base_path + mat_data_filename + '/validate/'
    validate_data_lists = os.listdir(validate_mat_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_mat_path + each_mat_name

        validate_mat_data_info = sio.loadmat(mat_path)
        validate_mat_data = validate_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_validate_mat_data = transform.resize(validate_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_validate_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])
        # one_hot_validate_mat_label = keras.utils.to_categorical(validate_mat_label, num_classes)

        for angle_num in range(aug_rotate_num):
            angle_step = 360/aug_rotate_num
            this_rotate_angle = angle_num*angle_step
            M = cv2.getRotationMatrix2D(((resize_width - 1) / 2, (resize_height - 1) / 2), this_rotate_angle, 1)
            rotated_mat_data = cv2.warpAffine(swap_resize_mat_data2, M, (resize_width, resize_height), borderMode=1)

            restore_mat_data1 = np.swapaxes(rotated_mat_data, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(validate_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_validate_mat_data = validate_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = validate_mat_data[:, :, 0, :, :]

            diff_validate_mat_data = diff_validate_mat_data + np.random.randn(1,1,polar_img_num-1,resize_width,resize_height)*0.002
            first_resized_polar_image = first_resized_polar_image + np.random.randn(1,1,resize_width,resize_height)*0.002

            if counter == 0:
                batch_diff_validate_data = diff_validate_mat_data
                batch_validate_label = validate_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
                batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'validate_data_label.h5', 'w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_mat_path = base_path + mat_data_filename + '/test/'
    test_data_lists = os.listdir(test_mat_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_mat_path + each_mat_name

        test_mat_data_info = sio.loadmat(mat_path)
        test_mat_data = test_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_test_mat_data = transform.resize(test_mat_data, (polar_img_num, resize_width, resize_height))

        test_mat_data = np.expand_dims(resize_test_mat_data, axis=0)
        test_mat_data = np.expand_dims(test_mat_data, axis=0)

        ext_first_resized_polar_image = np.expand_dims(test_mat_data[:, :, 0, :, :], axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_test_mat_data = test_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])
        # one_hot_test_mat_label = keras.utils.to_categorical(test_mat_label, num_classes)

        first_resized_polar_image = test_mat_data[:, :, 0, :, :]

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(aug_mat_path + 'test_data_label.h5', 'w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_end = 1

def flip_aug_and_save_to_h5(base_path, mat_data_filename, class_names):

    aug_mat_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'flip/'
    if not(os.path.exists(aug_mat_path)):
        os.makedirs(aug_mat_path)

    train_mat_path = base_path + mat_data_filename + '/train/'
    train_data_lists = os.listdir(train_mat_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_mat_path + each_mat_name

        train_mat_data_info = sio.loadmat(mat_path)
        train_mat_data = train_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_train_mat_data = transform.resize(train_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_train_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])

        train_mat_data = np.expand_dims(resize_train_mat_data, axis=0)  
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        if counter == 0:
            batch_diff_train_data = diff_train_mat_data
            batch_train_label = train_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
            counter = counter + 1
        else:
            batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
            batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate(
                (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,0)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0)  
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0)  
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,-1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
        train_mat_data = np.expand_dims(train_mat_data, axis=0)

        first_resized_polar_image = train_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_train_mat_data = train_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
        batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'train_data_label.h5', 'w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    validate_mat_path = base_path + mat_data_filename + '/validate/'
    validate_data_lists = os.listdir(validate_mat_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_mat_path + each_mat_name

        validate_mat_data_info = sio.loadmat(mat_path)
        validate_mat_data = validate_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_validate_mat_data = transform.resize(validate_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_validate_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])

        validate_mat_data = np.expand_dims(resize_validate_mat_data, axis=0) 
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        if counter == 0:
            batch_diff_validate_data = diff_validate_mat_data
            batch_validate_label = validate_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
            counter = counter + 1
        else:
            batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
            batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate(
                (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,0)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0) 
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)  
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

        fliped_swap_resize_mat_data2 = cv2.flip(swap_resize_mat_data2,-1)
        restore_mat_data1 = np.swapaxes(fliped_swap_resize_mat_data2, 0, 2)
        restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

        validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
        validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

        first_resized_polar_image = validate_mat_data[:, :, 0, :, :]
        ext_first_resized_polar_image = np.expand_dims(first_resized_polar_image, axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_validate_mat_data = validate_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
        batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
        batch_first_resized_polar_image = np.concatenate(
            (batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'validate_data_label.h5', 'w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_mat_path = base_path + mat_data_filename + '/test/'
    test_data_lists = os.listdir(test_mat_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_mat_path + each_mat_name

        test_mat_data_info = sio.loadmat(mat_path)
        test_mat_data = test_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_test_mat_data = transform.resize(test_mat_data, (polar_img_num, resize_width, resize_height))

        test_mat_data = np.expand_dims(resize_test_mat_data, axis=0)
        test_mat_data = np.expand_dims(test_mat_data, axis=0)

        ext_first_resized_polar_image = np.expand_dims(test_mat_data[:, :, 0, :, :], axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_test_mat_data = test_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])
        # one_hot_test_mat_label = keras.utils.to_categorical(test_mat_label, num_classes)

        first_resized_polar_image = test_mat_data[:, :, 0, :, :]

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(aug_mat_path + 'test_data_label.h5', 'w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_end = 1

def scale_aug_and_save_to_h5(base_path, mat_data_filename, class_names):

    aug_mat_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'scale/'
    if not(os.path.exists(aug_mat_path)):
        os.makedirs(aug_mat_path)
    base_scale_value = 1 - (aug_scale_num - 1) / 2 * aug_scale_step

    train_mat_path = base_path + mat_data_filename + '/train/'
    train_data_lists = os.listdir(train_mat_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_mat_path + each_mat_name

        train_mat_data_info = sio.loadmat(mat_path)
        train_mat_data = train_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_train_mat_data = transform.resize(train_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_train_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])
        # one_hot_train_mat_label = keras.utils.to_categorical(train_mat_label, num_classes)

        for scale_num in range(aug_scale_num):
            scale_value = base_scale_value + scale_num*aug_scale_step

            if scale_value < 1:   
                scaled_swap_resize_mat_data2 = transform.rescale(swap_resize_mat_data2, scale=scale_value)
                top = int((swap_resize_mat_data2.shape[0] - scaled_swap_resize_mat_data2.shape[0]) / 2)
                bottom = swap_resize_mat_data2.shape[0] - scaled_swap_resize_mat_data2.shape[0] - top
                left = int((swap_resize_mat_data2.shape[1] - scaled_swap_resize_mat_data2.shape[1]) / 2)
                right = swap_resize_mat_data2.shape[1] - scaled_swap_resize_mat_data2.shape[1] - left
                scaled_swap_resize_mat_data2 = cv2.copyMakeBorder(scaled_swap_resize_mat_data2, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)

            elif scale_value > 1:
                scaled_swap_resize_mat_data2 = transform.rescale(swap_resize_mat_data2, scale=scale_value)
                center_height = scaled_swap_resize_mat_data2.shape[0] / 2
                center_width = scaled_swap_resize_mat_data2.shape[1] / 2
                top = int(center_height - swap_resize_mat_data2.shape[0] / 2)
                bottom = top + swap_resize_mat_data2.shape[0]
                left = int(center_width - swap_resize_mat_data2.shape[1] / 2)
                right = left + swap_resize_mat_data2.shape[1]
                scaled_swap_resize_mat_data2 = scaled_swap_resize_mat_data2[top:bottom, left:right, :]

            else:
                scaled_swap_resize_mat_data2 = swap_resize_mat_data2

            # plt.imshow(swap_resize_mat_data2[:,:,0])
            # plt.show()
            # plt.imshow(scaled_swap_resize_mat_data2[:,:,0])
            # plt.show()

            restore_mat_data1 = np.swapaxes(scaled_swap_resize_mat_data2, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            train_mat_data = np.expand_dims(train_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(train_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_train_mat_data = train_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = train_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_train_data = diff_train_mat_data
                batch_train_label = train_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
                batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'train_data_label.h5', 'w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    validate_mat_path = base_path + mat_data_filename + '/validate/'
    validate_data_lists = os.listdir(validate_mat_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_mat_path + each_mat_name

        validate_mat_data_info = sio.loadmat(mat_path)
        validate_mat_data = validate_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_validate_mat_data = transform.resize(validate_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_validate_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])
        # one_hot_validate_mat_label = keras.utils.to_categorical(validate_mat_label, num_classes)

        for scale_num in range(aug_scale_num):
            scale_value = base_scale_value + scale_num*aug_scale_step

            if scale_value < 1: 
                scaled_swap_resize_mat_data2 = transform.rescale(swap_resize_mat_data2, scale=scale_value)
                top = int((swap_resize_mat_data2.shape[0] - scaled_swap_resize_mat_data2.shape[0]) / 2)
                bottom = swap_resize_mat_data2.shape[0] - scaled_swap_resize_mat_data2.shape[0] - top
                left = int((swap_resize_mat_data2.shape[1] - scaled_swap_resize_mat_data2.shape[1]) / 2)
                right = swap_resize_mat_data2.shape[1] - scaled_swap_resize_mat_data2.shape[1] - left
                scaled_swap_resize_mat_data2 = cv2.copyMakeBorder(scaled_swap_resize_mat_data2, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)

            elif scale_value > 1:
                scaled_swap_resize_mat_data2 = transform.rescale(swap_resize_mat_data2, scale=scale_value)
                center_height = scaled_swap_resize_mat_data2.shape[0] / 2
                center_width = scaled_swap_resize_mat_data2.shape[1] / 2
                top = int(center_height - swap_resize_mat_data2.shape[0] / 2)
                bottom = top + swap_resize_mat_data2.shape[0]
                left = int(center_width - swap_resize_mat_data2.shape[1] / 2)
                right = left + swap_resize_mat_data2.shape[1]
                scaled_swap_resize_mat_data2 = scaled_swap_resize_mat_data2[top:bottom, left:right, :]

            else:
                scaled_swap_resize_mat_data2 = swap_resize_mat_data2

            # plt.imshow(swap_resize_mat_data2[:,:,0])
            # plt.show()
            # plt.imshow(scaled_swap_resize_mat_data2[:,:,0])
            # plt.show()

            restore_mat_data1 = np.swapaxes(scaled_swap_resize_mat_data2, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(validate_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_validate_mat_data = validate_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = validate_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_validate_data = diff_validate_mat_data
                batch_validate_label = validate_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
                batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)


    f = h5py.File(aug_mat_path + 'validate_data_label.h5', 'w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_mat_path = base_path + mat_data_filename + '/test/'
    test_data_lists = os.listdir(test_mat_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_mat_path + each_mat_name

        test_mat_data_info = sio.loadmat(mat_path)
        test_mat_data = test_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_test_mat_data = transform.resize(test_mat_data, (polar_img_num, resize_width, resize_height))

        test_mat_data = np.expand_dims(resize_test_mat_data, axis=0)
        test_mat_data = np.expand_dims(test_mat_data, axis=0)

        ext_first_resized_polar_image = np.expand_dims(test_mat_data[:, :, 0, :, :], axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_test_mat_data = test_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])
        # one_hot_test_mat_label = keras.utils.to_categorical(test_mat_label, num_classes)

        first_resized_polar_image = test_mat_data[:, :, 0, :, :]

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(aug_mat_path + 'test_data_label.h5', 'w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_end = 1

def translation_aug_and_save_to_h5(base_path, mat_data_filename, class_names):

    aug_mat_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'translation/'
    if not(os.path.exists(aug_mat_path)):
        os.makedirs(aug_mat_path)
    base_translation_value = 0 - (aug_translation_num - 1) / 2 * aug_translation_step

    train_mat_path = base_path + mat_data_filename + '/train/'
    train_data_lists = os.listdir(train_mat_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_mat_path + each_mat_name

        train_mat_data_info = sio.loadmat(mat_path)
        train_mat_data = train_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_train_mat_data = transform.resize(train_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_train_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])
        # one_hot_train_mat_label = keras.utils.to_categorical(train_mat_label, num_classes)

        for translation_num in range(aug_translation_num):
            translation_value = base_translation_value + translation_num*aug_translation_step
            tx = translation_value
            ty = translation_value

            if translation_value != 0: 
                trans_M = np.float32([[1, 0, tx], [0, 1, ty]])
                height, width = swap_resize_mat_data2.shape[:2]
                translated_swap_resize_mat_data2 = cv2.warpAffine(swap_resize_mat_data2, trans_M, (width, height), borderMode=1)

            else:
                translated_swap_resize_mat_data2 = swap_resize_mat_data2

            # plt.imshow(swap_resize_mat_data2[:,:,0])
            # plt.show()
            # plt.imshow(translated_swap_resize_mat_data2[:,:,0])
            # plt.show()

            restore_mat_data1 = np.swapaxes(translated_swap_resize_mat_data2, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            train_mat_data = np.expand_dims(train_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(train_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_train_mat_data = train_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = train_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_train_data = diff_train_mat_data
                batch_train_label = train_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
                batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'train_data_label.h5', 'w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    validate_mat_path = base_path + mat_data_filename + '/validate/'
    validate_data_lists = os.listdir(validate_mat_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_mat_path + each_mat_name

        validate_mat_data_info = sio.loadmat(mat_path)
        validate_mat_data = validate_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_validate_mat_data = transform.resize(validate_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_validate_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])
        # one_hot_validate_mat_label = keras.utils.to_categorical(validate_mat_label, num_classes)

        for translation_num in range(aug_translation_num):
            translation_value = base_translation_value + translation_num*aug_translation_step
            tx = translation_value
            ty = translation_value

            if translation_value != 0: 
                trans_M = np.float32([[1, 0, tx], [0, 1, ty]])
                height, width = swap_resize_mat_data2.shape[:2]
                translated_swap_resize_mat_data2 = cv2.warpAffine(swap_resize_mat_data2, trans_M, (width, height), borderMode=1)

            else:
                translated_swap_resize_mat_data2 = swap_resize_mat_data2

            # plt.imshow(swap_resize_mat_data2[:,:,0])
            # plt.show()
            # plt.imshow(scaled_swap_resize_mat_data2[:,:,0])
            # plt.show()

            restore_mat_data1 = np.swapaxes(translated_swap_resize_mat_data2, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(validate_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_validate_mat_data = validate_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = validate_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_validate_data = diff_validate_mat_data
                batch_validate_label = validate_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
                batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)


    f = h5py.File(aug_mat_path + 'validate_data_label.h5', 'w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_mat_path = base_path + mat_data_filename + '/test/'
    test_data_lists = os.listdir(test_mat_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_mat_path + each_mat_name

        test_mat_data_info = sio.loadmat(mat_path)
        test_mat_data = test_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_test_mat_data = transform.resize(test_mat_data, (polar_img_num, resize_width, resize_height))

        test_mat_data = np.expand_dims(resize_test_mat_data, axis=0)
        test_mat_data = np.expand_dims(test_mat_data, axis=0)

        ext_first_resized_polar_image = np.expand_dims(test_mat_data[:, :, 0, :, :], axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_test_mat_data = test_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])
        # one_hot_test_mat_label = keras.utils.to_categorical(test_mat_label, num_classes)

        first_resized_polar_image = test_mat_data[:, :, 0, :, :]

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(aug_mat_path + 'test_data_label.h5', 'w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

def blur_aug_and_save_to_h5(base_path, mat_data_filename, class_names):

    aug_mat_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'blur/'
    if not(os.path.exists(aug_mat_path)):
        os.makedirs(aug_mat_path)
    base_blur_value = 0

    train_mat_path = base_path + mat_data_filename + '/train/'
    train_data_lists = os.listdir(train_mat_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_mat_path + each_mat_name

        train_mat_data_info = sio.loadmat(mat_path)
        train_mat_data = train_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_train_mat_data = transform.resize(train_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_train_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])
        # one_hot_train_mat_label = keras.utils.to_categorical(train_mat_label, num_classes)

        for blur_num in range(aug_blur_num):
            blur_value = base_blur_value + blur_num*aug_blur_step

            if blur_value != 0:
                blur_swap_resize_mat_data2 = cv2.GaussianBlur(swap_resize_mat_data2, blur_wsize, blur_value)
            else:
                blur_swap_resize_mat_data2 = swap_resize_mat_data2

            # plt.imshow(swap_resize_mat_data2[:,:,0])
            # plt.show()
            # plt.imshow(translated_swap_resize_mat_data2[:,:,0])
            # plt.show()

            restore_mat_data1 = np.swapaxes(blur_swap_resize_mat_data2, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            train_mat_data = np.expand_dims(train_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(train_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_train_mat_data = train_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = train_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_train_data = diff_train_mat_data
                batch_train_label = train_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
                batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'train_data_label.h5', 'w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    validate_mat_path = base_path + mat_data_filename + '/validate/'
    validate_data_lists = os.listdir(validate_mat_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_mat_path + each_mat_name

        validate_mat_data_info = sio.loadmat(mat_path)
        validate_mat_data = validate_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_validate_mat_data = transform.resize(validate_mat_data, (polar_img_num, resize_width, resize_height))

        swap_resize_mat_data1 = np.swapaxes(resize_validate_mat_data, 0, 2)
        swap_resize_mat_data2 = np.swapaxes(swap_resize_mat_data1, 0, 1)

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])
        # one_hot_validate_mat_label = keras.utils.to_categorical(validate_mat_label, num_classes)

        for blur_num in range(aug_blur_num):
            blur_value = base_blur_value + blur_num*aug_blur_step

            if blur_value != 0:
                blur_swap_resize_mat_data2 = cv2.GaussianBlur(swap_resize_mat_data2, blur_wsize, blur_value)
            else:
                blur_swap_resize_mat_data2 = swap_resize_mat_data2

            # plt.imshow(swap_resize_mat_data2[:,:,0])
            # plt.show()
            # plt.imshow(scaled_swap_resize_mat_data2[:,:,0])
            # plt.show()

            restore_mat_data1 = np.swapaxes(blur_swap_resize_mat_data2, 0, 2)
            restore_mat_data2 = np.swapaxes(restore_mat_data1, 1, 2)

            validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(validate_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_validate_mat_data = validate_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = validate_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_validate_data = diff_validate_mat_data
                batch_validate_label = validate_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
                batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)


    f = h5py.File(aug_mat_path + 'validate_data_label.h5', 'w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_mat_path = base_path + mat_data_filename + '/test/'
    test_data_lists = os.listdir(test_mat_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_mat_path + each_mat_name

        test_mat_data_info = sio.loadmat(mat_path)
        test_mat_data = test_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_test_mat_data = transform.resize(test_mat_data, (polar_img_num, resize_width, resize_height))

        test_mat_data = np.expand_dims(resize_test_mat_data, axis=0)
        test_mat_data = np.expand_dims(test_mat_data, axis=0)

        ext_first_resized_polar_image = np.expand_dims(test_mat_data[:, :, 0, :, :], axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_test_mat_data = test_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])
        # one_hot_test_mat_label = keras.utils.to_categorical(test_mat_label, num_classes)

        first_resized_polar_image = test_mat_data[:, :, 0, :, :]

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(aug_mat_path + 'test_data_label.h5', 'w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

def gnoised_aug_and_save_to_h5(base_path, mat_data_filename, class_names):

    aug_mat_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'gnoise/'
    if not(os.path.exists(aug_mat_path)):
        os.makedirs(aug_mat_path)
    base_gnoise_value = 0

    train_mat_path = base_path + mat_data_filename + '/train/'
    train_data_lists = os.listdir(train_mat_path)
    counter = 0
    for each_mat_name in train_data_lists:
        mat_path = train_mat_path + each_mat_name

        train_mat_data_info = sio.loadmat(mat_path)
        train_mat_data = train_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_train_mat_data = transform.resize(train_mat_data, (polar_img_num, resize_width, resize_height))

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            train_mat_label = 0
        elif cls_name == class_names[1]:
            train_mat_label = 1
        elif cls_name == class_names[2]:
            train_mat_label = 2
        else:
            train_mat_label = 3
        train_mat_label = np.array([train_mat_label])
        # one_hot_train_mat_label = keras.utils.to_categorical(train_mat_label, num_classes)

        for gnoise_num in range(aug_gnoise_num):
            gnoise_value = base_gnoise_value + gnoise_num*aug_gnoise_step
            restore_mat_data2 = resize_train_mat_data + np.random.randn(polar_img_num,resize_height,resize_width)*gnoise_value

            # plt.imshow(resize_train_mat_data[0,:,:])
            # plt.show()
            # plt.imshow(restore_mat_data2[0,:,:])
            # plt.show()

            train_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            train_mat_data = np.expand_dims(train_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(train_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_train_mat_data = train_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = train_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_train_data = diff_train_mat_data
                batch_train_label = train_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_train_data = np.concatenate((batch_diff_train_data, diff_train_mat_data), axis=0)
                batch_train_label = np.concatenate((batch_train_label, train_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'train_data_label.h5', 'w')
    f['data'] = batch_diff_train_data
    f['label'] = batch_train_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    validate_mat_path = base_path + mat_data_filename + '/validate/'
    validate_data_lists = os.listdir(validate_mat_path)
    counter = 0
    for each_mat_name in validate_data_lists:
        mat_path = validate_mat_path + each_mat_name

        validate_mat_data_info = sio.loadmat(mat_path)
        validate_mat_data = validate_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_validate_mat_data = transform.resize(validate_mat_data, (polar_img_num, resize_width, resize_height))

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            validate_mat_label = 0
        elif cls_name == class_names[1]:
            validate_mat_label = 1
        elif cls_name == class_names[2]:
            validate_mat_label = 2
        else:
            validate_mat_label = 3
        validate_mat_label = np.array([validate_mat_label])
        # one_hot_validate_mat_label = keras.utils.to_categorical(validate_mat_label, num_classes)

        for gnoise_num in range(aug_gnoise_num):
            gnoise_value = base_gnoise_value + gnoise_num*aug_gnoise_step
            restore_mat_data2 = resize_validate_mat_data + np.random.randn(polar_img_num,resize_height,resize_width)*gnoise_value

            validate_mat_data = np.expand_dims(restore_mat_data2, axis=0)
            validate_mat_data = np.expand_dims(validate_mat_data, axis=0)

            ext_first_resized_polar_image = np.expand_dims(validate_mat_data[:,:,0,:,:], axis=0)
            repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image,polar_img_num-1,axis = 2)
            diff_validate_mat_data = validate_mat_data[:,:,1:,:,:]-repeat_first_resized_polar_image

            first_resized_polar_image = validate_mat_data[:, :, 0, :, :]

            if counter == 0:
                batch_diff_validate_data = diff_validate_mat_data
                batch_validate_label = validate_mat_label
                batch_first_resized_polar_image = first_resized_polar_image
                counter = counter + 1
            else:
                batch_diff_validate_data = np.concatenate((batch_diff_validate_data, diff_validate_mat_data), axis=0)
                batch_validate_label = np.concatenate((batch_validate_label, validate_mat_label), axis=0)
                batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)

    f = h5py.File(aug_mat_path + 'validate_data_label.h5', 'w')
    f['data'] = batch_diff_validate_data
    f['label'] = batch_validate_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()

    test_mat_path = base_path + mat_data_filename + '/test/'
    test_data_lists = os.listdir(test_mat_path)
    counter = 0
    for each_mat_name in test_data_lists:
        mat_path = test_mat_path + each_mat_name

        test_mat_data_info = sio.loadmat(mat_path)
        test_mat_data = test_mat_data_info[mat_data_inner_name].astype('float64')/bit_max_value
        resize_test_mat_data = transform.resize(test_mat_data, (polar_img_num, resize_width, resize_height))

        test_mat_data = np.expand_dims(resize_test_mat_data, axis=0)
        test_mat_data = np.expand_dims(test_mat_data, axis=0)

        ext_first_resized_polar_image = np.expand_dims(test_mat_data[:, :, 0, :, :], axis=0)
        repeat_first_resized_polar_image = np.repeat(ext_first_resized_polar_image, polar_img_num - 1, axis=2)
        diff_test_mat_data = test_mat_data[:, :, 1:, :, :] - repeat_first_resized_polar_image

        cls_name = each_mat_name[0:7]
        if cls_name == class_names[0]:
            test_mat_label = 0
        elif cls_name == class_names[1]:
            test_mat_label = 1
        elif cls_name == class_names[2]:
            test_mat_label = 2
        else:
            test_mat_label = 3
        test_mat_label = np.array([test_mat_label])
        # one_hot_test_mat_label = keras.utils.to_categorical(test_mat_label, num_classes)

        first_resized_polar_image = test_mat_data[:, :, 0, :, :]

        if counter == 0:
            batch_diff_test_data = diff_test_mat_data
            batch_test_label = test_mat_label
            batch_first_resized_polar_image = first_resized_polar_image
        else:
            batch_diff_test_data = np.concatenate((batch_diff_test_data, diff_test_mat_data), axis=0)
            batch_test_label = np.concatenate((batch_test_label, test_mat_label), axis=0)
            batch_first_resized_polar_image = np.concatenate((batch_first_resized_polar_image, first_resized_polar_image), axis=0)
        counter = counter + 1

    f = h5py.File(aug_mat_path + 'test_data_label.h5', 'w')
    f['data'] = batch_diff_test_data
    f['label'] = batch_test_label
    f['first_image'] = batch_first_resized_polar_image
    f.close()


# learning rate schedule
def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

# 10.03, the best weight is 0.00007,0.5,30
def get_callbacks(initial_learning_rate = 0.00009, learning_rate_drop = 0.5, learning_rate_epochs = 30,
                  learning_rate_patience = 2, logging_file = "training.log", verbosity = 1,
                  early_stopping_patience = None):

    callbacks = list()

    model_checkpoint = ModelCheckpoint('weights/' + 'weight.{epoch:02d}-{val_softmax_acc:.4f}.hdf5',
                                       monitor='val_softmax_acc', verbose=0, save_best_only=False, mode='auto')
    callbacks.append(model_checkpoint)
    callbacks.append(CSVLogger(logging_file, append=True))
    if learning_rate_epochs:
        callbacks.append(LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate,
                                                       drop=learning_rate_drop, epochs_drop=learning_rate_epochs))
    else:
        callbacks.append(ReduceLROnPlateau(factor=learning_rate_drop, patience=learning_rate_patience,
                                           verbose=verbosity))                    
    if early_stopping_patience:
        callbacks.append(EarlyStopping(verbose=verbosity, patience=early_stopping_patience))     
    return callbacks


def main(overwrite = False):

    train_num = 109*(aug_rotate_num + aug_flip_num + aug_scale_num + aug_translation_num + aug_blur_num + aug_gnoise_num)
    validate_num = 54*(aug_rotate_num + aug_flip_num + aug_scale_num + aug_translation_num + aug_blur_num + aug_gnoise_num)
    test_num = 108
    class_names = ["0000231", "00bt474", "000mcf7", "sk-br-3"]
    base_path = '/media/DATA3/mdd/modified_polar_cell_data3/'

    mat_data_filename = 'random_select1/'
    mat_data_path = base_path + mat_data_filename

    # gnoise
    # aug_h5_data_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'gnoise/'
    # gnoised_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    # blur
    # aug_h5_data_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'blur/'
    # blur_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    # translation
    # aug_h5_data_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'translation/'
    # translation_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    # flip
    # aug_h5_data_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + 'flip/'
    # flip_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    # scale
    # aug_h5_data_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' -+ 'scale/'
    # scale_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    # rotate
    # aug_h5_data_path = base_path + 'aug_' + mat_data_filename + str(resize_width) + '_' + str(resize_height) + '_h5_' + str(aug_rotate_num) + '/'
    # rotate_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    # flip & rotate
    aug_h5_data_path = '/media/DATA3/mdd/modified_polar_cell_data3/compare_data/final_dataset/128_128_h5_rotate12_flip0_dualstream_firstcopy3/'
    # aug_h5_data_path = '/media/DATA3/mdd/modified_polar_cell_data3/compare_data/final_dataset/128_128_h5_rotate12_flip0_dualstream_firstcopy3/'
    flip_rotate_aug_and_save_to_h5(base_path, mat_data_filename, class_names)

    stop = 1

    # orig
    # aug_h5_data_path = base_path + 'diff_' + mat_data_filename[:-1] + '/' + str(resize_width) + '_' + str(resize_height) + '_h5/'
    # orig_save_mat_into_h5(base_path, mat_data_filename, h5_data_filename, class_names)

    # build the network
    concat_model,concat_test_model = unet_model_3d(first_input_shape, second_input_shape, 4, 1024)
    concat_model.summary()

    train_label, validate_label, test_label = my_get_train_validate_test_label(mat_data_path, class_names)
    train_batch_size = 54

    validate_batch_size = 27
    test_batch_size = 54

    second_train_generator, second_validate_generator, second_test_generator = get_second_training_validate_test_generators(aug_h5_data_path,train_batch_size,validate_batch_size,test_batch_size)

    train_or_test = 1
    if train_or_test == 1:
       concat_model.compile(optimizer = 'adam', loss=['sparse_categorical_crossentropy', lambda y_true, y_pred: y_pred],
                            loss_weights=[1., 0.01], metrics={'softmax':"accuracy"})
       # model_checkpoint = ModelCheckpoint('weights/' + 'weight.{epoch:02d}-{val_softmax_acc:.4f}.hdf5', monitor='val_softmax_acc', verbose = 0, save_best_only = True, mode='auto')

       concat_model.fit_generator(generator = second_train_generator, steps_per_epoch = int(train_num/train_batch_size), epochs = 150,
                                  validation_data = second_validate_generator,
                                  validation_steps = int(validate_num/validate_batch_size),
                                  callbacks = get_callbacks())
    elif train_or_test == 0:
       concat_test_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
       weight_filename = 'weights/'
       weight_list = os.listdir(weight_filename)
       for weight_name in weight_list:
           print(weight_name)

           concat_model.load_weights(weight_filename + weight_name)
           results2 = concat_test_model.predict_generator(second_test_generator, int(test_num / test_batch_size), verbose=1)
           result_index2 = np.argmax(results2, axis=1)
           result_accuracy2 = np.sum(result_index2 == test_label) / test_num

           print(result_accuracy2)

       test_end = 1
    test_end = 1

if __name__ == "__main__":

    np.random.seed(1337)
    main(overwrite=config["overwrite"])


