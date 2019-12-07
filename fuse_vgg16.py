#-*-coding:utf-8-*-
import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Lambda, Embedding
from keras.layers import Dropout,add,Reshape, GlobalAveragePooling2D,Multiply,Lambda,Add,GlobalAveragePooling3D,GlobalMaxPooling3D,Permute,Subtract
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet121

# K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

# center loss
def center_loss(x):
   numerator = K.sum(K.square(x[0] - x[1][:, 0, :]), 1, keepdims = True)
   return numerator


def fuse(input):
    data = input[0]
    pooling = input[1]
    pooling_expand1 = K.expand_dims(pooling, -2)
    pooling_expand2 = K.expand_dims(pooling_expand1, -2)
    pooling_expand3 = K.expand_dims(pooling_expand2, -2)
    pooling_repeat1 = K.repeat_elements(pooling_expand3, data.shape[1],  axis=1)
    pooling_repeat2 = K.repeat_elements(pooling_repeat1, data.shape[2], axis=2)
    pooling_repeat3 = K.repeat_elements(pooling_repeat2, data.shape[3], axis=3)

    fused_multiply = data*pooling_repeat3
    fused_sum = K.sum(fused_multiply,4)

    return fused_sum

def fuse_2d(input):
    data = input[0]
    pooling = input[1]
    pooling_expand1 = K.expand_dims(pooling,-1)
    pooling_expand2 = K.expand_dims(pooling_expand1,-1)
    pooling_repeat1 = K.repeat_elements(pooling_expand2,data.shape[2],axis=2)
    pooling_repeat2 = K.repeat_elements(pooling_repeat1, data.shape[3], axis=3)

    fused_multiply = data*pooling_repeat2
    # fused_sum = K.sum(fused_multiply,1)

    return fused_multiply


def unet_model_3d(first_input_shape, second_input_shape, nb_classes, feature_size):

    channel_first_first_input = Input(first_input_shape)
    first_input = Permute([2,3,4,1])(channel_first_first_input)

    first_conv_permute = Permute([4,2,3,1])(first_input)
    first_gpooling_0 = GlobalAveragePooling3D()(first_conv_permute)
    first_gpooling_dense_0 = Dense(units = 32, activation='linear')(first_gpooling_0)
    first_gpooling_dense_1_1 = Dense(units = 29, activation='sigmoid')(first_gpooling_dense_0)
    first_gpooling_fused_2 = Lambda(fuse)([first_conv_permute,first_gpooling_dense_1_1])

    first_conv_layer0 = Conv3D(8, (5, 5, 5), padding = 'same', activation='linear')(first_input)

    first_conv_permute = Permute([4,2,3,1])(first_conv_layer0)
    first_gpooling_0 = GlobalAveragePooling3D()(first_conv_permute)
    first_gpooling_dense_0 = Dense(units = 32, activation='linear')(first_gpooling_0)
    first_gpooling_dense_1_0 = Dense(units = 29, activation='sigmoid')(first_gpooling_dense_0)
    first_gpooling_fused_0 = Lambda(fuse)([first_conv_permute,first_gpooling_dense_1_0])

    first_conv_layer1 = Conv3D(8, (3, 3, 3), padding = 'same', activation='linear')(first_conv_layer0)

    first_conv_permute = Permute([4,2,3,1])(first_conv_layer1)
    first_gpooling_0 = GlobalAveragePooling3D()(first_conv_permute)
    first_gpooling_dense_0 = Dense(units = 32, activation='linear')(first_gpooling_0)
    first_gpooling_dense_1_1 = Dense(units = 29, activation='sigmoid')(first_gpooling_dense_0)
    first_gpooling_fused_1 = Lambda(fuse)([first_conv_permute,first_gpooling_dense_1_1])

    first_gpooling_add_0 = Add()([first_gpooling_fused_0, first_gpooling_fused_1,first_gpooling_fused_2])

    first_conv_layer2 = Conv2D(16, (3, 3), padding = 'same', activation='linear')(first_gpooling_add_0)
    first_pooling_layer1 = MaxPooling2D(pool_size=(2, 2))(first_conv_layer2)

    first_conv_layer3 = Conv2D(16, (3, 3), padding = 'same', activation='linear')(first_pooling_layer1)
    first_pooling_layer2 = MaxPooling2D(pool_size=(2, 2))(first_conv_layer3)

    first_conv_layer4 = Conv2D(16, (3, 3), padding = 'same', activation='linear')(first_pooling_layer2)
    first_pooling_layer3 = MaxPooling2D(pool_size=(2, 2),padding='same')(first_conv_layer4)

    first_flatten_layer1 = Flatten()(first_pooling_layer3)
    first_dense_layer1 = Dense(units = feature_size, activation='relu')(first_flatten_layer1)
    first_dense_layer2 = Dense(units=feature_size, activation='relu')(first_dense_layer1)

    base_model = VGG16(weights='imagenet',include_top=False,input_shape=[128,128,3])
    second_input = base_model.input
    block5_pool_output = base_model.output
    flatten = Flatten()(block5_pool_output)
    dense1 = Dense(4096,activation='relu')(flatten)
    dense2 = Dense(4096,activation='relu')(dense1)

    concat_layer = concatenate([first_dense_layer2, dense2],axis = 1)

    input_target = Input(shape=(1,)) 
    centers = Embedding(nb_classes, feature_size*5)(input_target)
    l2_loss = Lambda(center_loss, name='l2_loss')([concat_layer, centers])

    concat_result = Dense(units = nb_classes, activation = 'softmax',name = 'softmax')(concat_layer)
    concat_model = Model(inputs=[channel_first_first_input,second_input,input_target], outputs = [concat_result,l2_loss])
    concat_test_model = Model(inputs=[channel_first_first_input, second_input], outputs=concat_result)

    # return model_train, model_test, second_train_model
    return concat_model,concat_test_model


