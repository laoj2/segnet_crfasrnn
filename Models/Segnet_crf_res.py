from keras.layers.merge import Add
from keras.models import Model
from keras.layers import Reshape
from keras.layers import Input

from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
# from keras.regularizers import ActivityRegularizer
from keras import backend as K

from crfrnn_layer import CrfRnnLayer


def segnet_crf_res(nClasses, optimizer=None, input_height=360, input_width=480):
    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    img_input = Input(shape=(input_height, input_width,3))




    # encoder
    x = ZeroPadding2D(padding=(pad, pad))(img_input)
    x = Convolution2D(filter_size, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu') (x)
    l1 = x
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Convolution2D(128, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    l2 = x
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Convolution2D(256, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    l3 = x
    x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Convolution2D(512, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)
    l4 = x
    x = Activation('relu')(x)




    # decoder
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Convolution2D(512, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Add()([l4, x])
    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Convolution2D(256, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Add()([l3, x])
    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Convolution2D(128, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Add()([l2, x])
    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Convolution2D(filter_size, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)

    x = Add()([l1, x])
    x = Convolution2D(nClasses, (1, 1), padding='valid') (x)

    out = CrfRnnLayer(image_dims=(input_height, input_width),
                        num_classes=nClasses,
                        theta_alpha=160.,
                        theta_beta=3.,
                        theta_gamma=3.,
                        num_iterations=5,
                        name='crfrnn')([x, img_input])
#    out = x
    a = Model(inputs=img_input, outputs=out)

    model = []
    a.outputHeight = a.output_shape[1]
    a.outputWidth = a.output_shape[2]

    out = Reshape((a.outputHeight * a.outputWidth, nClasses), input_shape=(nClasses, a.outputHeight, a.outputWidth))(out)
    out = Activation('softmax')(out)
#    if not optimizer is None:
#        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    model = Model(inputs=img_input, outputs=out)
    model.outputHeight = a.outputHeight
    model.outputWidth = a.outputWidth

    return model
