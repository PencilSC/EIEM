import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import os
from keras import optimizers
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from keras.models import Model
from keras.layers import Dropout, Lambda, Input, average, Reshape, UpSampling2D, Multiply,Concatenate
from keras.layers import Conv2D, MaxPool2D, Conv2DTranspose, GlobalAveragePooling2D, Flatten, Dense, Add, \
     AveragePooling2D, Conv1D, GlobalMaxPooling2D, Activation
from keras.layers import ZeroPadding2D, Cropping2D, BatchNormalization, MaxPooling2D
from keras.layers import TimeDistributed, ConvLSTM2D
from keras import backend as K
from keras import losses
from keras.utils.np_utils import to_categorical

from keras.utils import multi_gpu_model

import numpy as np
from keras import regularizers
import SimpleITK as sitk
import cv2

import tensorflow as tf
from keras.layers import Bidirectional
from keras.layers import Permute

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
zweight = './model_logs/**.h5'

# 将余弦退火（带周期性重启）与Adamax优化器结合，并加入预热期
class CosineAnnealingScheduler(LearningRateSchedule):
    def __init__(self, lr_max, lr_min, T_max, T_warmup, T_mul):
        super(CosineAnnealingScheduler, self).__init__()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.T_mul = T_mul

    def __call__(self, step):
        if step < self.T_warmup:
            lr = self.lr_min + (self.lr_max - self.lr_min) * step / self.T_warmup
        else:
            if (step - self.T_warmup) % self.T_max == 0 and step > self.T_warmup: # step == self.T_warmup 时，%得0，但不能*mul
                self.T_max *= self.T_mul
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * ((step - self.T_warmup) % self.T_max) / self.T_max))

        # print('Global Step && Learning Rate :  {} && {}'.format(step, lr))

        return lr

def crop(tensors):
    '''
    :param tensors: List of two tensors, the second tensor having larger spatial dims
    :return:
    '''
    h_dims, w_dims = [], []
    for t in tensors:
        b, h, w, d = K.get_variable_shape( t )
        h_dims.append( h )
        w_dims.append( w )
    crop_h, crop_w = (h_dims[1] - h_dims[0]), (w_dims[1] - w_dims[0])
    rem_h = crop_h % 2
    rem_w = crop_w % 2
    crop_h_dims = (crop_h // 2, crop_h // 2 + rem_h)
    crop_w_dims = (crop_w // 2, crop_w // 2 + rem_w)
    cropped = Cropping2D( cropping=(crop_h_dims, crop_w_dims) )( tensors[1] )
    return cropped

def w_losses(bc1or0 = False, dice1or0 = False, ssim1or0 = False,  
              w_in_bc = 1.0, 
              w_bc = 1.0, w_dc = 1.0, w_ss = 1.0):
    def w_loss(y_true, y_pred):

        bc_loss = 0.0
        if bc1or0:
            # black-[1,0] atrium-[0,1] [负类,正类]
            y_pred1 = 1-y_pred
            ya = y_true*tf.log(tf.maximum(y_pred, 1e-8))
            yb = (1-y_true)*tf.log(tf.maximum(y_pred1, 1e-8))
            weight1 = w_in_bc/(w_in_bc+1.0)
            weight2 = 1.0-weight1
            bc_loss = -tf.reduce_mean(weight1*ya+weight2*yb)
        

        dice_loss = 0.0
        if dice1or0:
            eps = 1e-8
            intersection = tf.reduce_sum(y_true * y_pred) + eps
            summation = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + eps
            dice_loss = 1.0 - (2.0 * intersection/summation)


        ssim_loss = 0.0
        if ssim1or0:
            ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        return w_bc*bc_loss + w_dc*dice_loss + w_ss*ssim_loss
    return w_loss

def Active_Contour_Loss(y_true, y_pred): 
    #y_pred = K.cast(y_pred, dtype = 'float64')

    """
    lenth term
    """
    y_true=tf.reshape(y_true,(16,1,256,256))
    y_pred=tf.reshape(y_pred,(16,1,256,256))

    x = y_pred[:,:,1:,:] - y_pred[:,:,:-1,:] # horizontal and vertical directions 
    y = y_pred[:,:,:,1:] - y_pred[:,:,:,:-1]

    delta_x = x[:,:,1:,:-2]**2
    delta_y = y[:,:,:-2,1:]**2
    delta_u = K.abs(delta_x + delta_y) 

    epsilon = 0.00000001 # where is a parameter to avoid square root is zero in practice.
    w = 1
    lenth = w * K.sum(K.sqrt(delta_u + epsilon)) # equ.(11) in the paper

    """
    region term
    """
    C_1 = np.ones((256, 256))
    C_2 = np.zeros((256, 256))

    region_in = K.abs(K.sum( y_pred[:,0,:,:] * ((y_true[:,0,:,:] - C_1)**2) ) ) # equ.(12) in the paper
    region_out = K.abs(K.sum( (1-y_pred[:,0,:,:]) * ((y_true[:,0,:,:] - C_2)**2) )) # equ.(12) in the paper

    lambdaP = 1 # lambda parameter could be various.
    
    loss =  lenth + lambdaP * (region_in + region_out) 

    return loss

# m: input
# dim: the num of channel
# res: controls the res connection
# drop: controls the dropout layer
# initpara: initial parameters
def convblock_bn2(m, dim, layername, res=0, drop=0.5, **kwargs):
    n = Conv2D(filters=dim, name= layername + '_conv1', **kwargs)(m)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    n = Dropout(drop)(n) if drop else n
    n = Conv2D(filters=dim, name= layername + '_conv2', **kwargs)(n)
    n = BatchNormalization(momentum=0.95, epsilon=0.001)(n)
    return Add()([m, n]) if res else n

def convblock_bn1(m, dim, layername, res=0, drop=0.5, **kwargs):
    n = Conv2D(filters=dim, name= layername + '_conv1', **kwargs)(m)
    n = Conv2D(filters=dim, name= layername + '_conv2', **kwargs)(n)
    n = BatchNormalization()(n)
    n = Dropout(drop)(n) if drop else n
    return Add()([m, n]) if res else n

def convblock_bn0(m, dim, layername, res=0, drop=0.5, **kwargs):
    n = Conv2D(filters=dim, name= layername + '_conv1', **kwargs)(m)
    n = Dropout(drop)(n) if drop else n
    n = Conv2D(filters=dim, name= layername + '_conv2', **kwargs)(n)
    return Add()([m, n]) if res else n


# (-1)-Conv2D, 2-Bidirectional(ConvLSTM2D)
# filters1, name1, filters2, name2, input_layer
def zup(za, f1, n1, f2, n2, zlayer):
    if za == -1:
        zuplayer = Conv2D(filters=f1, kernel_size=1, padding='same', activation='relu',
            name=n1)(Conv2DTranspose(filters=f2, kernel_size=3, strides=2, padding='same',
                name=n2)(zlayer))
    if za == 0:
        zuplayer = TimeDistributed(Conv2D(filters=f1, kernel_size=1, padding='same', activation='relu',
            name=n1))(TimeDistributed(Conv2DTranspose(filters=f2, kernel_size=3, strides=2, padding='same',
                name=n2))(zlayer))
    if za == 1:
        zuplayer = ConvLSTM2D(filters=f1, kernel_size=1, padding='same', activation='relu',
            name=n1, return_sequences=True)(TimeDistributed(Conv2DTranspose(filters=f2, kernel_size=3,
                strides=2, padding='same', name=n2))(zlayer))
    if za == 2:
        zuplayer = Bidirectional(ConvLSTM2D(filters=f1, kernel_size=1, padding='same', activation='relu',
            name=n1, return_sequences=True))(TimeDistributed(Conv2DTranspose(filters=f2, kernel_size=3,
                strides=2, padding='same', name=n2))(zlayer))
    return zuplayer


def channel_attention(input_feature, ratio=8):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature._keras_shape[channel_axis]

    # 折扣率，用来节省开销
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return Multiply()([input_feature, cbam_feature])


def unet(input_shape, num_classes, lr_scheduler, pool=True, weights=None):
    '''initialization'''
    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',  # Xavier均匀初始化
        #kernel_initializer='he_normal',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,  # 施加在输出上的正则项
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,  # 权值是否更新
    )

    num_classes = num_classes
    data = Input(shape=input_shape, dtype='float', name='data')
    
    print('unet - input_shape')
    print(data.shape)




    # encoder
    enconv1 = convblock_bn2(data, dim=32, layername='block1', **kwargs)




    upedge1 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(data)
    upedge1 = channel_attention(upedge1)
    upedge1 = convblock_bn2(upedge1, dim=32, layername='upedge1', **kwargs)



    
    pool1 = MaxPooling2D(pool_size=3, strides=2,padding='same',name='pool1')(enconv1) if pool \
        else Conv2D(filters=32, strides=2, name='pool1')(enconv1)




    enconv2 = convblock_bn0(pool1, dim=64, layername='block2', **kwargs)




    edge2 = channel_attention(pool1)
    edge2 = convblock_bn2(edge2, dim=64, layername='edge2', **kwargs)
    upedge2 = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu',
            name='upedge2conv')(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                name='upedge2')(edge2))




    pool2 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool2')(enconv2) if pool \
        else Conv2D(filters=64, strides=2, name='pool2')(enconv2)




    enconv3 = convblock_bn0(pool2, dim=128, layername='block3', **kwargs)




    edge3 = channel_attention(pool2)
    edge3 = convblock_bn2(edge3, dim=128, layername='edge3', **kwargs)
    upedge3 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',
            name='upedge31conv')(Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                name='upedge31')(edge3))
    upedge3 = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu',
            name='upedge32conv')(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                name='upedge32')(upedge3))




    pool3 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool3')(enconv3) if pool \
        else Conv2D(filters=128, strides=2, name='pool3')(enconv3)




    enconv4 = convblock_bn0(pool3, dim=256, layername='block4', **kwargs)




    edge4 = channel_attention(pool3)
    edge4 = convblock_bn2(edge4, dim=256, layername='edge4', **kwargs)
    upedge4 = Conv2D(filters=128, kernel_size=1, padding='same', activation='relu',
            name='upedge41conv')(Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same',
                name='upedge41')(edge4))
    upedge4 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',
            name='upedge42conv')(Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                name='upedge42')(upedge4))
    upedge4 = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu',
            name='upedge43conv')(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                name='upedge43')(upedge4))




    pool4 = MaxPooling2D(pool_size=3, strides=2, padding='same', name='pool4')(enconv4) if pool \
        else Conv2D(filters=256, strides=2, name='pool4')(enconv4)




    enconv5 = convblock_bn0(pool4, dim=512, layername='block5notl', **kwargs)
    #print('enconv5.shape')
    #print(enconv5.shape)




    edge5 = channel_attention(pool4)
    edge5 = convblock_bn2(edge5, dim=512, layername='edge5', **kwargs)
    upedge5 = Conv2D(filters=256, kernel_size=1, padding='same', activation='relu',
            name='upedge51conv')(Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding='same',
                name='upedge51')(edge5))
    upedge5 = Conv2D(filters=128, kernel_size=1, padding='same', activation='relu',
            name='upedge52conv')(Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same',
                name='upedge52')(upedge5))
    upedge5 = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu',
            name='upedge53conv')(Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same',
                name='upedge53')(upedge5))
    upedge5 = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu',
            name='upedge54conv')(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same',
                name='upedge54')(upedge5))


    '''upedge1 = channel_attention(upedge1)
    upedge2 = channel_attention(upedge2)
    upedge3 = channel_attention(upedge3)
    upedge4 = channel_attention(upedge4)
    upedge5 = channel_attention(upedge5)'''


    medge1 = Concatenate()([upedge2,upedge1])
    medge2 = Concatenate()([upedge3,medge1])
    medge3 = Concatenate()([upedge4,medge2])
    medge4 = Concatenate()([upedge5,medge3])


    e_conv = Conv2D(filters=num_classes, kernel_size=1, padding='same', activation='relu',
        name='e_conv')(medge4)


    predictions1 = Conv2D(filters=num_classes, kernel_size=1, activation='sigmoid', padding='same',
        name='predictions1')(e_conv)
    print('unet - predictions1')
    print(predictions1.shape)



    enconv5 = Concatenate()([enconv5, edge5])
    # decoder
    up1 = zup(-1, 256, 'up1', 512, 'trans1', enconv5)
    merge1 = Concatenate()([up1, enconv4, edge4])
    deconv6 = convblock_bn0(merge1, dim=256, layername='deconv6', **kwargs)

    up2 = zup(-1, 128, 'up2', 256, 'trans2', deconv6)
    merge2 = Concatenate()([up2, enconv3, edge3])
    deconv7 = convblock_bn0(merge2, dim=128, layername='deconv7', **kwargs)

    up3 = zup(-1, 64, 'up3', 128, 'trans3', deconv7)
    merge3 = Concatenate()([up3, enconv2, edge2])
    deconv8 = convblock_bn0(merge3, dim=64, layername='deconv8', **kwargs)

    up4 = zup(-1, 32, 'up4', 64, 'trans4', deconv8)
    merge4 = Concatenate()([up4, enconv1, upedge1])
    deconv9 = convblock_bn2(merge4, dim=16, layername='deconv9', **kwargs)



    conv10 = Conv2D(filters=num_classes, kernel_size=1, padding='same', activation='relu',
        name='conv10')(deconv9)



    conv10 = Concatenate()([conv10,e_conv])



    predictions = Conv2D(filters=num_classes, kernel_size=1, activation='sigmoid', padding='same',
        name='predictions')(conv10)


    model = Model(inputs=data, outputs= [predictions, predictions1])

    '''
    if weights is None:
        model.load_weights(zweight, by_name=True)
    '''
    if weights is not None:
        model.load_weights(weights,by_name=True)
    
    sgd = optimizers.Adamax(lr=lr_scheduler, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


    '''bc1or0 = False, dice1or0 = False, ssim1or0 = False, 
       w_in_bc = 1.0, 
       w_bc = 1.0, w_dc = 1.0, w_ss = 1.0'''

    model.compile(optimizer=sgd, 
        loss={
        'predictions':w_losses(True,False,False),
        'predictions1':w_losses(True,False,False)
        },
        loss_weights={
        'predictions': 0.5,
        'predictions1': 0.5
        },
        metrics=['accuracy'])

    return model

if __name__ == '__main__':
    model = unet((256, 256, 1), 3, 0.001, pool=True, weights=None)