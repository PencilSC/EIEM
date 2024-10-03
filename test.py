# -*- coding: utf-8 -*-
rootC = '240324_111947_1'
range_from = 0
range_to = 10
rate = 0.5
lcc0or1 = 1

pre_name = '_pre.nrrd'
if lcc0or1 == 1:
    pre_name = '_pre_lcc.nrrd'

import numpy as np
fname_log = 'logs/' + rootC.split('_')[0] + '_' +rootC.split('_')[1] + 'vallog_1.txt'
ff=open(fname_log,"r", encoding = 'utf-8',errors='ignore')
counti = 0
arr2 = np.zeros((2,200))
for line in ff:
    arr2[0][counti] = float(line.split(' ')[-1][0:-9])#score
    arr2[1][counti] = float(line.split('[')[1].split(',')[0])#loss
    counti = counti+1
    if counti == range_to:
        break
score_max = np.max(arr2[0][range_from:range_to])
loss_min = np.min(arr2[1][range_from:range_to])
print(score_max)
print(loss_min)
ffnum = np.where(arr2[0]==score_max) 
print(ffnum)
ffnum = str(int(ffnum[0][0])+1)

rootD = ffnum
root00 = './data/test/'
root01 = rootC + '_epoch_' + rootD
rootA = root00 + 'LAUNet' + root01
rootB = root00 + root01

from skimage import io
import os,re
import cv2
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#import numpy as np
from datapre import *
from network import unet
from keras import backend as K
from keras.models import Model
from shutil import copyfile
import SimpleITK as sitk
from SimpleITK import ReadImage, GetArrayFromImage

import surface_distance as surfdist

# lcc
from skimage.measure import label
import nrrd

import warnings
warnings.filterwarnings("ignore")

DATA_PATH = './data'

ONLINE_IMG_PATH = os.path.join(DATA_PATH, './test/images')
ONLINE_CONTOUR_PATH = os.path.join(DATA_PATH, './test/labels')
ONLINE_EDGE_PATH = os.path.join(DATA_PATH, './test/edges')

import datetime
ztime = str(datetime.datetime.now()).split('.')[0]
mini_batch_size = 16
time_steps = 8

def dice_coef(img1, img2):
    intersection = np.sum(img1 * img2)
    summation = np.sum(img1) + np.sum(img2)
    dice = (2.0 * intersection+1.0) / (summation+1.0)
    return dice

def jaccard(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    return intersection / union

def create_submission(contours, data_path, mask_path, edge_path, weights):
    weights = weights
    crop_size = 300
    input_size = 256
    #time_steps = 28
    lr = 0.01
    print(data_path,mask_path,edge_path)
    images, masks, masks_orginal, edges = export_all4_contours(contours, data_path, mask_path, edge_path, crop_size)
    # images: 总time_steps数,time_steps,256,256,1
    '''
    images =  images.reshape(images.shape[0],images.shape[1],images.shape[2],images.shape[3], 1)
    masks = masks.reshape(masks.shape[0],masks.shape[1],masks.shape[2],masks.shape[3], 1)
    input_shape = (None, input_size, input_size, 1)
    '''
    input_shape = (input_size, input_size, 1)
    num_classes = 1
    model = unet(input_shape, num_classes, lr, weights=weights)
    '''
    #中间层特征输出
    model_inter = Model(inputs=model.input, outputs=model.get_layer('pool2').output)
    inter_mask = model_inter.predict(images)
    for _ in range(0, len(contours)):
        # plt.imshow的输入应为二维的
        print(contours[_])
        se2_img = inter_mask[_, :, :, :]
        # 通过通道c来对子图进行排序
        w, h, c = se2_img.shape
        fig, axes = plt.subplots(math.ceil(math.sqrt(float(c))), round(math.sqrt(float(c))),figsize=(math.sqrt(float(c))+1,math.sqrt(float(c))+1))
        # 调整子图之间的间隔
        fig.subplots_adjust(wspace=0.,hspace=0.)
        #获取子图坐标系
        axs = axes.ravel()
        #关闭所有子图坐标系显示
        for j in range(len(axs)):
            axs[j].axis('off')
        #显示子图
        for i in range(c):
            ax = axs[i]
            ax.imshow(se2_img[:, :, i])
        #plt.show()
        fig.savefig('inter_output/pool2+50_'+ contours[_],dpi=600)
        plt.close()
    '''
    #pred_masks = model.predict(images, batch_size=mini_batch_size, verbose=1)
    # pred_masks: 总time_steps数,time_steps,256,256,1
    pred_masks2 = model.predict(images, batch_size=mini_batch_size, verbose=1)
    pred_masks = pred_masks2[0]
    dice = []
    '''time_step的两层嵌套
    for idx, ctr in enumerate(contours):
        for idx1,ctr1 in enumerate(contours[idx]):
        
            # 直接使用预测结果计算dice值，shape=(300,300)
            img, mask = images[idx][idx1], masks[idx][idx1]
            tmp = pred_masks[idx][idx1]

            assert img.shape == tmp.shape, 'Shape of prediction does not match'
            lll = np.where(tmp > 0.5, 1, 0).astype('float64')
            dice_local = dice_coef(mask, lll)
            print('%s %f'% (contours[idx][idx1], dice_local))
            dice.append(dice_local)

            #存储图像
            #cv2.imwrite(os.path.join(save_dir, '%s-img.png' % (contours[idx])), img)
            cv2.imwrite(os.path.join(save_dir, '%s_pre.png' % (contours[idx][idx1][0:-4])), lll*255)
            cv2.imwrite(os.path.join(save_dir, '%s_label.png' % (contours[idx][idx1][0:-4])), mask*255)
    '''
    #masks = export_masks(contours, mask_path, crop_size)
    for idx, ctr in enumerate(contours):
        # 直接使用预测结果计算dice值，shape=(300,300)
        img, mask = images[idx], masks_orginal[idx]
        tmp = pred_masks[idx]
        target_size = (300, 300)
        tmp = cv2.resize(tmp, target_size, interpolation=cv2.INTER_LINEAR)
        if tmp.ndim < 3:
            tmp = tmp[..., np.newaxis]
        #assert img.shape == tmp.shape, 'Shape of prediction does not match'
        lll = np.where(tmp > rate, 1, 0).astype('float64')
        dice_local = dice_coef(mask, lll)
        dice.append(dice_local)
        #存储图像
        cv2.imwrite(os.path.join(save_dir, '%s-pre.png' % (contours[idx][0:-4])), lll*255)
        cv2.imwrite(os.path.join(save_dir, '%s-label.png' % (contours[idx][0:-4])), mask*255)

    mean_dice = np.mean(dice)
    print('mean dice_coef :{}, std dicd:{}'.format(mean_dice, np.std(dice)))
    return mean_dice

# move2.py
def move2():
    # copyfile(src, dst)
    # src:源文件
    # dst：目标文件
    sfrom = rootA
    sto1 = rootB
    zstrs = [['.*label.png$','/2dlabel'],['.*pre.png$','/2dpre']]
    for spng in zstrs:
        for fname in os.listdir(sfrom):
            if re.search(spng[0],fname):
                ffrom = os.path.join(sfrom, fname)
                sto = sto1 + spng[1]
                if not os.path.exists(sto):
                    os.makedirs(sto)
                nrrd_filename = os.path.join(sto, fname)
                copyfile(ffrom, nrrd_filename)
    return 1

# png2nrrd.py
def png2nrrd(sort_by=1):

    num = rootB
    zstr = ['pre','label']
    for zs in zstr:
        filepath='./'+num+'/2d'+zs
        files=os.listdir(filepath)
        files.sort()

        if sort_by==1:
            # 61_0-pre/label.png
            files.sort(key = lambda x: [int(x.split('_')[0]),int(x.split('_')[1].split('-')[0])])
            print('sort by 1')
        if sort_by==2:
            # 61_0_0_pre/label.png
            files.sort(key = lambda x: [int(x.split('_')[0]),int(x.split('_')[2])])
            print('sort by 2')
        
        im3d=np.zeros(shape=(88,io.imread(os.path.join(filepath,files[0])).shape[0],
              io.imread(os.path.join(filepath,files[0])).shape[1]), dtype='uint16')
        count=0
        zb = '80'
        for file_ in files:
            za = file_.split('_')[0]
            if zb == za:
                im2d=io.imread(os.path.join(filepath,file_))
                im3d[count]=im2d
                count+=1
            else:
                filename1 = './'+str(num)+'/3d'+zs+'/'
                if not os.path.exists(filename1):
                    os.makedirs(filename1)
                filename = os.path.join(filename1, '%s_%s.nrrd' % (zb,zs))
                out = sitk.GetImageFromArray(im3d)
                sitk.WriteImage(out, filename)
                im3d=np.zeros(shape=(88,io.imread(os.path.join(filepath,files[0])).shape[0],
                      io.imread(os.path.join(filepath,files[0])).shape[1]), dtype='uint16')
                zb = za
                count=0
                im2d=io.imread(os.path.join(filepath,file_))
                im3d[count]=im2d
                count+=1

        filename = os.path.join(filename1, '%s_%s.nrrd' % (zb,zs))
        out = sitk.GetImageFromArray(im3d)
        sitk.WriteImage(out, filename)
        '''
        nrrd.write('./2.nrrd',im3d)
        '''
        # print(im3d.shape)
        # print(im3d.dtype)
    return 1

# copy
def copy():

    sfrom1 = rootB
    sto = rootB

    zstrs = ['/3dlabel','/3dpre']
    for snrrd in zstrs:
        sfrom = sfrom1 + snrrd
        for fname in os.listdir(sfrom):
            ffrom = os.path.join(sfrom, fname)
            ffto = os.path.join(sto, fname)
            copyfile(ffrom, ffto)
    return 1

def read_nrrd(file_name):
    '''
    读取nrrd体数据文件
    :param file_name:nrrd文件路径
    :return:nd-array，(z,y,x)
    '''
    img = ReadImage(file_name)
    return GetArrayFromImage(img)

def jaccard_coef_example(y_true, y_pred):
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    return intersection / union

def jaccard_coef(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - np.sum(y_true * y_pred)
    return intersection / union

def hausdorff_dist(y_true, y_pred):
    quality=dict()
    labelTrue=sitk.GetImageFromArray(y_true, isVector=False)
    labelPred=sitk.GetImageFromArray(y_pred, isVector=False)

    SurfaceDistanceMeasures1=sitk.HausdorffDistanceImageFilter()
    SurfaceDistanceMeasures1.Execute(labelTrue>0.5,labelPred>0.5)
    quality["HausdorffDistance"]=SurfaceDistanceMeasures1.GetHausdorffDistance()
    quality["AverageHausdorffDistance"]=SurfaceDistanceMeasures1.GetAverageHausdorffDistance()

    OverlapMeasures=sitk.LabelOverlapMeasuresImageFilter()
    OverlapMeasures.Execute(labelTrue>0.5,labelPred>0.5)
    quality["DiceCoefficient"] = OverlapMeasures.GetDiceCoefficient()
    quality["JaccardCoefficient"] = OverlapMeasures.GetJaccardCoefficient()
    quality["VolumeSimilarity"] = OverlapMeasures.GetVolumeSimilarity()
    quality["FalseNegativeError"] = OverlapMeasures.GetFalseNegativeError()
    quality["FalsePositiveError"] = OverlapMeasures.GetFalsePositiveError()

    return quality["HausdorffDistance"]

# evaluate_v.py
def evaluate_v(coefficient_i, coefficient_str):

    path = rootB +'/'
    coefficient = []
    for num in range(80, 100):

        label_data = read_nrrd(path + str(num) + '_label.nrrd')
        #print(label_data.shape)
        pre_data = read_nrrd(path + str(num) + pre_name)
        #print(pre_data.shape)
        label_data1 = np.where(label_data > 0, 1, 0).astype('float64')
        pre_data1 = np.where(pre_data > 0, 1, 0).astype('float64')
        label_data2 = np.where(label_data > 0, True, False)
        pre_data2 = np.where(pre_data > 0, True, False)

        if coefficient_i==1:
            coefficient_v = dice_coef(label_data1, pre_data1)
        if coefficient_i==2:
            coefficient_v = jaccard_coef(label_data1, pre_data1)
        if coefficient_i==3:
            coefficient_v = hausdorff_dist(label_data1, pre_data1)
        if coefficient_i==4:
            surface_distances = surfdist.compute_surface_distances(label_data2, pre_data2, spacing_mm=(0.625, 0.625, 0.625))
            coefficient_v = surfdist.compute_robust_hausdorff(surface_distances, 95)
        if coefficient_i==5:
            surface_distances = surfdist.compute_surface_distances(label_data2, pre_data2, spacing_mm=(0.625, 0.625, 0.625))
            coefficient_v = surfdist.compute_average_surface_distance(surface_distances)
            coefficient_v = np.mean(coefficient_v)

        print('{} - {} : {}'.format(num, coefficient_str, coefficient_v))
        
        coefficient.append(coefficient_v)
        
    mean_coefficient = np.mean(coefficient)
    print('mean {} : {} , std dicd : {}'.format(coefficient_str, mean_coefficient, np.std(coefficient)))



    testlog2 = testdir+'3D_'+rootC+'.txt'
    test_res2 = open(testlog2, 'a+')
    test_res2.write('%s : %s\n' % (coefficient_str, [round(x, 6) for x in coefficient]))
    test_res2.close()



    return mean_coefficient, np.std(coefficient)

def largestConnectComponent(bw_img, ):
    '''
    compute largest Connect component of a binary image
    Parameters:
    ---
    bw_img: ndarray
        binary image
    Returns:
    ---
    lcc: ndarray
        largest connect component.
    Example:
    ---
        >>> lcc = largestConnectComponent(bw_img)
    '''
    labeled_img, num = label(bw_img, background=0, return_num=True)    
    # plt.figure(), plt.imshow(labeled_img, 'gray')
    max_label = 0
    max_num = 0
    for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    return lcc

def dolcc():
    for zstr in range(80,100):
        pre_data = read_nrrd(rootB+'/'+str(zstr)+'_pre.nrrd')
        print(pre_data.shape)

        lcc = largestConnectComponent(pre_data)
        im3d = lcc.astype('uint16')
        print(im3d.shape)

        strto = rootB+'/'+str(zstr)+'_pre_lcc.nrrd'
        out = sitk.GetImageFromArray(im3d)
        sitk.WriteImage(out, strto)
        #a = nrrd.write(strto,im3d)

        pre_m = read_nrrd(strto)
        print(pre_m.shape)
        print('finish')

if __name__== '__main__':

    zz = 'LAUNet'+root01

    ztime1 = str(datetime.datetime.now())
    weight = './model_logs/' + zz + '.h5'
    save_dir = './data/test/' + zz
    if  not os.path.exists(save_dir):#如果path不存在
        os.makedirs(save_dir)
    print('\nProcessing online contours...')
    online_ctrs = get_all_images(ONLINE_IMG_PATH, shuffle=True)#[:4]
    # batch_size*time_steps维图像名
    ztime2 = str(datetime.datetime.now())

    md = create_submission(online_ctrs, ONLINE_IMG_PATH, ONLINE_CONTOUR_PATH, ONLINE_EDGE_PATH, weight)
    
    testdir = './results/'
    if not os.path.exists(testdir):
        os.makedirs(testdir)
    testlog1 = testdir+rootC+'.txt'
    test_res1 = open(testlog1, 'a+')
    test_res1.write('%s : %s\n' % (root01, md))
    test_res1.close()

    ztime3 = str(datetime.datetime.now())
    print('ztime1')
    print(ztime1)
    print('ztime2')
    print(ztime2)
    print('ztime3')
    print(ztime3)

    print('\nAll done.')

    move2()
    png2nrrd()
    copy()
    if lcc0or1 == 1:
        dolcc()

    testlog2 = testdir+'3D_'+rootC+'.txt'
    test_res2 = open(testlog2, 'a+')

    res11,res12 = evaluate_v(1, 'DiceCoefficient')
    res21,res22 = evaluate_v(2, 'JaccardCoefficient')
    res31,res32 = evaluate_v(3, 'HausdorffDistance')
    res41,res42 = evaluate_v(4, 'hd_dist_95')
    res51,res52 = evaluate_v(5, 'avg_surf_dist')

    test_res2.write('%s\n' % root01)
    if lcc0or1 == 1:
        test_res2.write('with lcc\n')
    if lcc0or1 == 0:
        test_res2.write('without lcc\n')
    test_res2.write('range from %s to %s : %s\n' % (range_from, range_to, ffnum))
    test_res2.write('rate : %s\n' % rate)
    test_res2.write('      DiceCoefficient : %s\n          +/-  %s\n' % (res11, res12))
    test_res2.write('      JaccardCoefficient : %s\n          +/-  %s\n' % (res21, res22))
    test_res2.write('      HausdorffDistance : %s\n          +/-  %s\n' % (res31, res32))
    test_res2.write('      hd_dist_95 : %s\n          +/-  %s\n' % (res41, res42))
    test_res2.write('      avg_surf_dist : %s\n          +/-  %s\n\n' % (res51, res52))
    test_res2.close()
    print('range from ' +str(range_from)+ ' to '+str(range_to)+' : '+ffnum)