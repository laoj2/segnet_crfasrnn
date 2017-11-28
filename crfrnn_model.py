"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation, Reshape, Permute
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, \
    Dropout, Conv2DTranspose, Cropping2D, Add, BatchNormalization
from crfrnn_layer import CrfRnnLayer


def get_crfrnn_model_def (nClasses , optimizer=None , input_height=360, input_width=480 ):
    """ Returns Keras CRN-RNN model definition.

    Currently, only 500 x 500 images are supported. However, one can get this to
    work with different image sizes by adjusting the parameters of the Cropping2D layers
    below.
    """

    channels, height, weight = 3, input_height, input_width

    # Input
    input_shape = (height, weight, 3)
    img_input = Input(shape=input_shape)

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2


    # Add plenty of zero padding
    x = ZeroPadding2D(padding=(pad, pad))(img_input)


    # VGG-16 convolution block 1
    x = Conv2D(filter_size, (kernel, kernel), padding='valid', name='conv1_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((pool_size, pool_size), name='pool1')(x)


    # VGG-16 convolution block 1
    x = Conv2D(128, (kernel, kernel), padding='valid', name='conv1_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((pool_size, pool_size),  name='pool2')(x)

    # VGG-16 convolution block 1
    x = Conv2D(256, (kernel, kernel), padding='valid', name='conv1_3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((pool_size, pool_size),  name='pool3')(x)
    pool3 = x

    # VGG-16 convolution block 1
    x = Conv2D(512, (kernel, kernel), padding='valid', name='conv1_4')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((pool_size, pool_size),  name='pool4')(x)
    pool4 = x

    #decoder
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(512, (kernel, kernel), padding='valid', name='conv2_1')(x)
    x = BatchNormalization()(x)


    x = UpSampling2D((pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(256, (kernel, kernel), padding='valid', name='conv2_2')(x)
    x = BatchNormalization()(x)



    x = UpSampling2D((pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(128, (kernel, kernel), padding='valid', name='conv2_3')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((pool_size, pool_size))(x)
    x = ZeroPadding2D(padding=(pad, pad))(x)
    x = Conv2D(filter_size, (kernel, kernel), padding='valid', name='conv2_4')(x)
    x = BatchNormalization()(x)




    x = Conv2D(nClasses, (1, 1), padding='valid', name='conv3_1')(x)
    #x = Conv2D(100,(kernel,kernel),padding='valid')(x)

    #out_height = x.shape[1]
    #out_width = x.shape[2]

    #x = Reshape((nClasses,32*32), input_shape=(32, 32, nClasses))(x)

    #x = Permute((2,1))(x)

    #x = Activation('softmax')(x)
    print x
    #x = UpSampling2D(size=(4,4))(x)


    output = CrfRnnLayer(image_dims=(32, 32),
                         num_classes=nClasses,
                         theta_alpha=160.,
                         theta_beta=3.,
                         theta_gamma=3.,
                         num_iterations=10,
                         name='crfrnn')([x, img_input])

    # Build the model
    model = Model(img_input, output, name='crfrnn_net')
    model.outputHeight = 32
    model.outputWidth = 32
    return model
