import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers import *


def Conv3DWithBN(x, filters, ksize, strides, name, padding='same', dilation_rate=1, center=True, scale=True, decay=0.99):
    x = Conv3D(filters=filters, kernel_size=ksize, strides=strides, padding=padding, dilation_rate=dilation_rate,
                        use_bias=False, kernel_initializer='he_normal', name=name+'_conv')(x)
    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn')(x)
    x = Activation('relu', name=name+'_relu')(x)
    return x


def Conv2DWithBN(x, filters, ksize, strides, name, padding='same', dilation_rate=1, center=True, scale=True, decay=0.99):
    x = Conv2D(filters=filters, kernel_size=ksize, strides=strides, padding=padding, dilation_rate=dilation_rate,
                        use_bias=False, kernel_initializer='he_normal', name=name+'_conv')(x)
    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn')(x)
    x = Activation('relu', name=name+'_relu')(x)
    return x


def Conv1DWithBN(x, filters, ksize, strides, name, padding='same', dilation_rate=1, center=True, scale=True, decay=0.99):
    x = Conv1D(filters=filters, kernel_size=ksize, strides=strides, padding=padding, dilation_rate=dilation_rate,
               use_bias=False, kernel_initializer='he_normal', name=name+'_conv')(x)
    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn')(x)
    x = Activation('relu', name=name+'_relu')(x)
    return x


def DenseWithBN(x, units, name, kernel_regularizer=None, center=True, scale=True, decay=0.99):
    x = Dense(units=units, use_bias=False, kernel_regularizer=kernel_regularizer, name=name+'_weight')(x)
    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bias')(x)
    x = Activation('relu', name=name+'_relu')(x)
    return x


def ResNetUnit2D(x, filters, ksize, name, end=False, center=True, scale=True, decay=0.99):
    identity = x

    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_1')(x)
    x = Activation('relu', name=name+'_relu_1')(x)
    x = Conv2D(filters, kernel_size=ksize, strides=1, padding='same', kernel_initializer='he_normal', name=name+'_conv_1')(x)

    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_2')(x)
    x = Activation('relu', name=name+'_relu_2')(x)
    x = Conv2D(filters, kernel_size=ksize, strides=1, padding='same', kernel_initializer='he_normal', name=name+'_conv2')(x)

    x = add([x, identity])
    if end:
        x = BatchNormalization(center=center, scale=scale, momentum=decay)(x)
        x = Activation('relu')(x)
    return x


def ResNetUnitIncreasingDims2D(x, filters, ksize, strides, name, begin=False, center=True, scale=True, decay=0.99):
    identity = x

    if not begin:
        x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_1')(x)
        x = Activation('relu', name=name+'_relu_1')(x)
    x = Conv2D(filters, kernel_size=ksize, strides=strides[0], padding='same', kernel_initializer='he_normal', use_bias=False, name=name+'_conv_1')(x)

    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_2')(x)
    x = Activation('relu', name=name+'_relu_2')(x)
    x = Conv2D(filters, kernel_size=ksize, strides=strides[1], padding='same', kernel_initializer='he_normal', use_bias=False, name=name+'_conv_2')(x)

    identity = Conv2D(filters, kernel_size=1, strides=strides[0], padding='same', kernel_initializer='he_normal', name=name+'_conv_identity')(identity)
    x = add([x, identity])
    return x


def ResNetUnit1D(x, filters, ksize, name, end=False, center=True, scale=True, decay=0.99):
    identity = x

    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_1')(x)
    x = Activation('relu', name=name+'_relu_1')(x)
    x = Conv1D(filters, kernel_size=ksize, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False, name=name+'_conv_1')(x)

    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_2')(x)
    x = Activation('relu', name=name+'_relu_2')(x)
    x = Conv1D(filters, kernel_size=ksize, strides=1, padding='same', kernel_initializer='he_normal', use_bias=False, name=name+'_conv_2')(x)

    x = add([x, identity])
    if end:
        x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_3')(x)
        x = Activation('relu', name=name+'_relu_3')(x)
    return x


def ResNetUnitIncreasingDims1D(x, filters, ksize, strides, name, begin=False, center=True, scale=True, decay=0.99):
    '''
    ResNet unit without BottleNeck. 2 layers
    :param x:
    :param filters:
    :param ksize:
    :param strides: list with 2 elements, stride for each layer
    :param begin:
    :return:
    '''

    identity = x

    if not begin:
        x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_1')(x)
        x = Activation('relu', name=name+'_relu_1')(x)
    x = Conv1D(filters, kernel_size=ksize, strides=strides[0], padding='same', kernel_initializer='he_normal', use_bias=False, name=name+'_conv_1')(x)

    x = BatchNormalization(center=center, scale=scale, momentum=decay, name=name+'_bn_2')(x)
    x = Activation('relu', name=name+'_relu_2')(x)
    x = Conv1D(filters, kernel_size=ksize, strides=strides[1], padding='same', kernel_initializer='he_normal', use_bias=False, name=name+'_conv_2')(x)

    identity = Conv1D(filters, kernel_size=1, strides=strides[0], padding='same', kernel_initializer='he_normal', name=name+'_conv_identity')(identity)
    x = add([x, identity])
    return x


def ContextualAvgPooling(x, ksizes, strides):
    '''
    Concatenate input with pooling result
    :param x:
    :param ksizes:
    :param strides:
    :return:
    '''
    out = None
    for ks in ksizes:
        x_pooled = AvgPool1D(pool_size=ks, strides=strides, padding='same')
        if out is None:
            out = x_pooled
        else:
            out = concatenate([out, x_pooled], axis=-1)
    x = concatenate([x, out], axis=-1)
    return x


def ContextualAtrousConv1D(x, filters, ksize, strides, dilation_rates, name):
    """
    Retrieve contextual information with Atrous Convolution
    :param x:
    :param filters:
    :param ksize:
    :param strides:
    :param dilation_rates:
    :return:
    """
    concat = x
    for dr in dilation_rates:
        x_atrous = Conv1DWithBN(x, filters=filters, ksize=ksize, strides=strides, dilation_rate=dr, name=name+'a_conv_s'+str(dr))
        concat = concatenate([concat, x_atrous], axis=-1)
    concat = Conv1DWithBN(concat, filters=256, ksize=ksize, strides=strides, name=name+'atrou_post_conv1')
    return concat


def ContextualAtrousConv3D(x, filters, ksize, strides, dilation_rates, name):
    """
    Retrieve contextual information with Atrous Convolution
    :param x:
    :param filters:
    :param ksize:
    :param strides:
    :param dilation_rates:
    :return:
    """
    concat = None
    for dr in dilation_rates:
        x_atrous = Conv3DWithBN(x, filters=filters, ksize=ksize, strides=strides, dilation_rate=dr, name=name+'a_conv_s'+str(dr))
        if concat is None:
            concat = x_atrous
        else:
            concat = concatenate([concat, x_atrous], axis=-1)
    concat = Conv3DWithBN(concat, filters=filters, ksize=ksize, strides=strides, name=name+'atrou_post_conv1')
    return concat


def SharedAtrousConv1D(x, SharedConvs, PostConv):
    concat = x
    for SharedConv in SharedConvs:
        x_atrous = SharedConv(x)
        x_atrous = BatchNormalization()(x_atrous)
        x_atrous = Activation('relu')(x_atrous)
        concat = concatenate([concat, x_atrous], axis=-1)
    concat = PostConv(concat)
    concat = BatchNormalization()(concat)
    concat = Activation('relu')(concat)
    return concat


def densenet_block3d(x, k, rep):
    dense_input = x
    for i in range(rep):
        x_dense = Conv3D(filters=k, kernel_size=3, strides=1, padding='same', activation='relu')(dense_input)
        dense_input = concatenate([dense_input, x_dense])
    return dense_input


def DenseNetTransit(x, rate=1, name=None):
    if rate != 1:
        out_features = x.get_shape().as_list()[-1] * rate
        x = BatchNormalization(center=True, scale=True, name=name + '_bn')(x)
        x = Activation('relu', name=name + '_relu')(x)
        x = Conv3D(filters=out_features, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                   use_bias=False, name=name + '_conv')(x)
    x = AveragePooling3D(pool_size=2, strides=2, padding='same')(x)
    return x


def DenseNetUnit3D(x, growth_rate, ksize, rep, bn_decay=0.99, name=None):
    for i in range(rep):
        concat = x
        x = BatchNormalization(center=True, scale=True, momentum=bn_decay, name=name+'_bn_rep_'+str(i))(x)
        x = Activation('relu')(x)
        x = Conv3D(filters=growth_rate, kernel_size=ksize, padding='same',
                   kernel_initializer='glorot_normal', use_bias=False, name=name+'_conv_rep_'+str(i))(x)
        x = concatenate([concat, x])
    return x


class BilinearUpsampling3D(Layer):
    """
    Wrapping 1D BilinearUpsamling as a Keras layer
    Input: 3D Tensor (batch, dim, channels)
    """
    def __init__(self, size, **kwargs):
        self.size = size
        super(BilinearUpsampling3D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BilinearUpsampling3D,self).build(input_shape)

    def call(self, x, mask=None):
        x = tf.expand_dims(x, axis=2)
        x = tf.image.resize_bilinear(x, [self.size, 1])
        x = tf.squeeze(x, axis=2)
        return x

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.size, input_shape[2])


# def SharedAtrousConv1D(x, SharedConvs, PostConv):
#     concat = None
#     for SharedConv in SharedConvs:
#         x_atrous = SharedConv(x)
#         x_atrous = BatchNormalization()(x_atrous)
#         x_atrous = Activation('relu')(x_atrous)
#         if concat is None:
#             concat = x_atrous
#         else:
#             concat = concatenate([concat, x_atrous], axis=-1)
#     concat = PostConv(concat)
#     concat = BatchNormalization()(concat)
#     concat = Activation('relu')(concat)
#     return concat


class BilinearUpsampling1D(Layer):
    """
    Wrapping 1D BilinearUpsamling as a Keras layer
    Input: 3D Tensor (batch, dim, channels)
    """
    def __init__(self, size, **kwargs):
        self.size = size
        super(BilinearUpsampling1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BilinearUpsampling1D,self).build(input_shape)

    def call(self, x, mask=None):
        x = tf.expand_dims(x, axis=2)
        x = tf.image.resize_bilinear(x, [self.size, 1])
        x = tf.squeeze(x, axis=2)
        return x

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.size, input_shape[2])






