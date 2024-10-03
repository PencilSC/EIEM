import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import os, re
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import TensorBoard
import tensorflow as tf
import numpy as np 
from sklearn.model_selection import KFold
from keras.callbacks import ModelCheckpoint
# ndarray打印省略问题 
np.set_printoptions(threshold=np.inf)

import datetime
ztime = str(datetime.datetime.now()).split('.')[0]
ztime = ztime.replace('-','')
ztime = ztime.replace(' ','_')
ztime = ztime.replace(':','')
ztime = ztime[2:]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from network import unet, CosineAnnealingScheduler
from datapre import *
mini_batch_size = 16
# 数据路径
DATA_PATH = './data/'
# 迭代次数
epochs = 200
lr_1st = 0.001

# 设定参数
lr_max = 0.001
lr_min = 0.0007
T_max = 88*60//mini_batch_size  # 整个数据集的批次数
T_warmup = epochs*0*T_max
T_mul = 2  # 重启周期的乘子，如果设定为2，那么每个周期的长度将是前一个周期的两倍

from PIL import Image

seed = 4567
# 数据路径
TRAIN_LABEL_PATH = os.path.join(DATA_PATH, 'train/labels')
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train/images')
TRAIN_EDGE_PATH = os.path.join(DATA_PATH, 'train/edges')

VAL_LABEL_PATH = TRAIN_LABEL_PATH
VAL_IMG_PATH = TRAIN_IMG_PATH
VAL_EDGE_PATH = TRAIN_EDGE_PATH

ONLINE_LABEL_PATH = os.path.join(DATA_PATH, 'test/labels')
ONLINE_IMG_PATH = os.path.join(DATA_PATH, 'test/images')
ONLINE_EDGE_PATH = os.path.join(DATA_PATH, 'test/edges')

from keras.callbacks import LearningRateScheduler
import math

# 学习率下降策略
def lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5):
    lrate = base_lr * (1.0 - (curr_iter / float(max_iter))) ** power
    K.set_value(model.optimizer.lr, lrate)
    return K.eval(model.optimizer.lr)

def dice_coef_online(img1, img2):
    eps = 1e-8
    intersection = np.sum(img1 * img2)
    summation = np.sum(img1) + np.sum(img2)
    dice = (2.0 * intersection+eps) / (summation+eps)
    #print(intersection)
    #print(summation)
    return dice

def online_pre_multiloss(model, online_ctrs, img, masks, state='test'):
    pred_masksn = model.predict(img, batch_size=mini_batch_size, verbose=1)
    pred_masks = pred_masksn[0]
    dice_list = []
    dice_test = []
    for idx, ctr in enumerate(online_ctrs):
        #pred_masks[idx] = np.where(pred_masks[idx] > 0.5, 1., 0.)
        dice = dice_coef_online(masks[idx], pred_masks[idx])
        #print('dice')
        #print(dice)
        dice_list.append(dice)
    md = np.mean(dice_list)
    print('md :{}, std dicd:{}'.format(md,np.std(dice_list)))
    return md

def train_dice(mask, pre):
    dice = []
    for gt,p in zip(mask, pre):
        dice_local = dice_coef_online(gt, p)
        #print(dice_local)
        dice.append(dice_local)
    return dice

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


if __name__ == '__main__':

    # 中心裁剪至300*300，再resize到256*256
    crop_size = 300
    input_size = 256
    print('0. Mapping ground truth contours to images in train...')

    # input_shape 为(time_steps, map_height, map_width, channels)
    input_shape = (input_size, input_size,1)
    num_classes = 1
    # 数据扩增参数
    kwargs = dict(
        rotation_range=20,
        zoom_range=0.1,
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #horizontal_flip=True,
        #vertical_flip=True,
    )


    fold_no = 1

    arr = np.arange(0,80,1,int) # 生成0-79的array(数组)
    kfold = KFold(n_splits=5, shuffle=True)

    
    aaaa = np.arange(0,60,1,int)
    bbbb = np.arange(60,80,1,int)
    for train, dev in kfold.split(arr):

        train = aaaa
        dev = bbbb

        if not os.path.exists('./logs'):
             os.mkdir('./logs')
        logs_ztime = './logs/' + ztime
        iterlog = logs_ztime + 'iterlog.txt'

        iter_res = open(iterlog, 'a+')
        iter_res.write('iter -- %s :%s %s\n' % (str(fold_no), train, dev))
        iter_res.close()

        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'1. Training for fold {fold_no} ...')

        train_ctrs = get_all_images_i(TRAIN_IMG_PATH, train, shuffle=True)#[:32]
        print('2.1. Done reading training set')
        dev_ctrs = get_all_images_i(VAL_IMG_PATH, dev, shuffle=True)#[:32]
        print('2.2. Done reading validating set')

        print('3.1. Building Train dataset ...')
        img_train, mask_train, edge_train = export_all3_contours(train_ctrs,
                                                    TRAIN_IMG_PATH,
                                                    TRAIN_LABEL_PATH,
                                                    TRAIN_EDGE_PATH,
                                                    crop_size=crop_size)
        print(img_train.shape)
        print('3.2. Building Dev dataset ...')
        img_dev, mask_dev, edge_dev = export_all3_contours(dev_ctrs,
                                                VAL_IMG_PATH,
                                                VAL_LABEL_PATH,
                                                VAL_EDGE_PATH,
                                                crop_size=crop_size)
        print(img_dev.shape)
        print('3.3. Building done, model')


        '''print('原始CropAndResize后的train data(img) - max and min : ', np.max(img_train), np.min(img_train))
        # 使用unique函数获取唯一值
        unique_values = np.unique(img_train)
        print("数量：", len(unique_values))
        print('原始CropAndResize后的train data(mask) - max and min : ', np.max(mask_train), np.min(mask_train))
        unique_values = np.unique(mask_train)
        print("数量：", len(unique_values))
        print('原始CropAndResize后的dev data(mask) - max and min : ', np.max(mask_dev), np.min(mask_dev))
        unique_values = np.unique(mask_dev)
        print("数量：", len(unique_values))'''


        global_step = 0  # 维护全局的训练步数
        # 初始化学习率调度器
        lr_scheduler = CosineAnnealingScheduler(lr_max, lr_min, T_max, T_warmup, T_mul) 
        model = unet(input_shape, num_classes, lr_scheduler(global_step), pool=True, weights=None)
        # model,model_org = unet(input_shape, num_classes, lr_1st, pool=True, weights=None)
        

        # data augementation
        print('4. data augumentation....')
        image_datagen = ImageDataGenerator(**kwargs)
        mask_datagen = ImageDataGenerator(**kwargs)
        edge_datagen = ImageDataGenerator(**kwargs)
        print('4.1. img generating....')
        image_generator = image_datagen.flow(img_train, shuffle=True,
                                             batch_size=mini_batch_size, seed=seed)
        print('4.2. mask generating....')
        mask_generator = mask_datagen.flow(mask_train, shuffle=True,
                                           batch_size=mini_batch_size, seed=seed)
        print('4.3. edge generating....')
        edge_generator = edge_datagen.flow(edge_train, shuffle=True,
                                           batch_size=mini_batch_size, seed=seed)
        print('4.4. zip....')
        train_generator = zip(image_generator, mask_generator, edge_generator)
        

        max_iter = (len(train_ctrs) // mini_batch_size) * epochs
        curr_iter = 0
        base_lr = K.eval(model.optimizer.lr)


        print('5. training')

        if not os.path.exists('./graphs'):
             os.mkdir('./graphs')
        log_path = './graphs/graph'+ztime+'_'+str(fold_no)
        callback = TensorBoard(log_path)
        callback.set_model(model)

        train_names = ['train_loss', 'train_acc', 'train_dice']
        val_names = ['val_loss', 'val_acc', 'val_dice']



        for e in range(epochs):
            print('\n6. Main Epoch {:d}'.format(e + 1))
            lr_current = K.eval(model.optimizer.lr)
            print('   Learning rate: {:6f}\n'.format(lr_current))
            train_res = []
            train_dice_list = []

            for iteration in range(len(img_train) // mini_batch_size):
                img, mask, edge = next(train_generator)


                '''print('augmentation后的train data(img) - max and min : ', np.max(img), np.min(img))
                unique_values = np.unique(img)
                print("数量：", len(unique_values))
                print('augmentation后的train data(mask) - max and min : ', np.max(mask), np.min(mask))
                unique_values = np.unique(mask)
                print("数量：", len(unique_values))'''

                # 二值化处理，将插值后的浮点数值转换为整数值
                mask = np.where(mask >= 0.5, 1.0, 0.0)

                '''print('界0.5后的train data(mask) - max and min : ', np.max(mask), np.min(mask))
                unique_values = np.unique(mask)
                print("数量：", len(unique_values))'''

                save_dirr = './linshi1'
                for idx, ctr in enumerate(mask):
                    #print(mask[idx].shape)                    
                    #print(edge[idx].shape)

                    mk = cv2.resize(mask[idx], (256, 256))
                    
                    # 使用腐蚀操作检测边缘
                    kernel = np.ones((3, 3), np.uint8)
                    eroded = cv2.erode(mk, kernel, iterations=1)
                    eg = mk - eroded

                    eg = eg[..., np.newaxis]

                    edge[idx]=eg

                '''print('erode后的train data(edge) - max and min : ', np.max(edge), np.min(edge))
                unique_values = np.unique(edge)
                print("数量：", len(unique_values))'''

                #edge = np.where(edge == 255, 1.0, 0.0)

                '''print('0/1后的train data(edge) - max and min : ', np.max(edge), np.min(edge))
                unique_values = np.unique(edge)
                print("数量：", len(unique_values))'''

                '''for idx, ctr in enumerate(img):
                    cv2.imwrite(os.path.join(save_dirr, '%s_%s-img.png' % (global_step,idx)), img[idx])
                    cv2.imwrite(os.path.join(save_dirr, '%s_%s-mask.png' % (global_step,idx)), mask[idx]*255)
                    cv2.imwrite(os.path.join(save_dirr, '%s_%s-edge.png' % (global_step,idx)), edge[idx]*255)'''


                res = model.train_on_batch(img, [mask, edge])
                global_step += 1  # 更新全局步数
                #K.set_value(model.optimizer.lr, lr_scheduler(global_step)) # 更新、赋值给下一个step用
                # print(K.eval(model.optimizer.lr))
                curr_iter += 1
                lrate = lr_poly_decay(model, base_lr, curr_iter, max_iter, power=0.5)


                train_res.append(res)
                pren = model.predict(img)
                pre = pren[0]


                #print('pre.shape')
                #print(pre.shape)
                td = train_dice(mask,pre)
                train_dice_list.append(td)


            train_res = np.asarray(train_res)
            train_res = np.mean(train_res, axis=0).round(decimals=5) # 按列求平均，保留decimals位小数
            train_dice_list = np.asarray(train_dice_list)
            # print('train_dice_list')
            # print(train_dice_list)
            train_dice_list = np.mean(train_dice_list)
            # print('train_dice_list')
            # print(train_dice_list)


            # graph1: tensorboard --logdir ./graph --port 8999
            step_la1 = e + 1
            write_log(callback, train_names, [train_res[0], train_res[3], train_dice_list], step_la1)


            trainlog = logs_ztime + 'trainlog'+'_'+str(fold_no)+'.txt'
            vallog = logs_ztime + 'vallog'+'_'+str(fold_no)+'.txt'

            train_rec = open(trainlog, 'a+')
            train_rec.write('epochs -- %s :%s %s %s\n' % (str(e+1), lr_current, train_res, train_dice_list))
            train_rec.close()
            print('Train result :\nloss: {}     dice: {}'.format(train_res, train_dice_list))


            print('\nEvaluating dev set ...')


            mask_dev = np.where(mask_dev >= 0.5, 1.0, 0.0)

            '''print('界0.5后的dev data(mask) - max and min : ', np.max(mask_dev), np.min(mask_dev))
            unique_values = np.unique(mask_dev)
            print("数量：", len(unique_values))'''


            save_dirr = './linshi2'
            for idx, ctr in enumerate(mask_dev):
                #print(mask_dev[idx].shape)                    
                #print(edge_dev[idx].shape)

                mk = cv2.resize(mask_dev[idx], (256, 256))
                '''mk= mk.astype(np.uint8)
                eg=cv2.Canny(mk*255,85,170)'''

                kernel = np.ones((3, 3), np.uint8)
                eroded = cv2.erode(mk, kernel, iterations=1)
                eg = mk - eroded

                eg = eg[..., np.newaxis]

                edge_dev[idx]=eg

            '''print('erode后的dev data(edge) - max and min : ', np.max(edge_dev), np.min(edge_dev))
            unique_values = np.unique(edge_dev)
            print("数量：", len(unique_values))'''

            #edge_dev = np.where(edge_dev == 255, 1.0, 0.0)

            '''print('0/1后的dev data(edge) - max and min : ', np.max(edge_dev), np.min(edge_dev))
            unique_values = np.unique(edge_dev)
            print("数量：", len(unique_values))'''

            '''for idx, ctr in enumerate(img_dev):
                cv2.imwrite(os.path.join(save_dirr, '%s_%s-img.png' % (global_step,idx)), img_dev[idx])
                cv2.imwrite(os.path.join(save_dirr, '%s_%s-mask.png' % (global_step,idx)), mask_dev[idx]*255)
                cv2.imwrite(os.path.join(save_dirr, '%s_%s-edge.png' % (global_step,idx)), edge_dev[idx]*255)'''


            val_res = model.evaluate(img_dev, [mask_dev, edge_dev], batch_size=mini_batch_size)
            val_res = [round(num, 5) for num in val_res]
            result_dice = online_pre_multiloss(model, dev_ctrs, img_dev, mask_dev, state='validation')

            val_rec = open(vallog, 'a+')
            val_rec.write('epochs -- %s :%s %s\n' % (str(e+1), val_res, result_dice))
            val_rec.close()
            print('\nDev result :\nloss: {}     dice: {}'.format(val_res, result_dice))


            # 模型存储，每1个epoch存储一次模型
            za = 'LAUNet' + ztime + '_' + str(fold_no)
            save_file = '_'.join([za, 'epoch', str(e + 1)]) + '.h5'
            if not os.path.exists('model_logs'):
                os.makedirs('model_logs')
            save_path = os.path.join('model_logs', save_file)
            
            if (e+1) > 0:
                print('\nSaving model weights to {}'.format(save_path))
                model.save_weights(save_path)

            # graph2
            step_la2 = e + 1
            write_log(callback, val_names, [val_res[0], val_res[3], result_dice], step_la2)



        fold_no = fold_no + 1
        break