from keras.models import Model
from keras.layers import Reshape
from keras.layers import Input

from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D, Conv2DTranspose
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D

def segnet_transposed(nClasses, optimizer=None, input_height=360, input_width=480):
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
    x = Conv2DTranspose(512, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)

#    x = Add()([l4, x])
    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2DTranspose(256, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)

 #   x = Add()([l3, x])
    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2DTranspose(128, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2DTranspose(filter_size, (kernel, kernel), padding='valid')(x)
    x = BatchNormalization()(x)

   # x = Add()([l1, x])
    x = Conv2DTranspose(nClasses, (1, 1), padding='valid') (x)

    out = x
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

