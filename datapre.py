import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import os, re
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import cv2
import numpy as np
from sklearn import preprocessing as pre

from sklearn.preprocessing import MinMaxScaler

def center_crop(ndarray, crop_size):
    '''Input ndarray is of rank 3 (height, width, depth).
    Argument crop_size is an integer for square cropping only.
    Performs padding and center cropping to a specified size.
    '''
    h, w, d = ndarray.shape
    if crop_size == 0:
        raise ValueError('argument crop_size must be non-zero integer')
    if any([dim < crop_size for dim in (h, w)]):
        # zero pad along each (h, w) dimension before center cropping
        pad_h = (crop_size - h) if (h < crop_size) else 0
        pad_w = (crop_size - w) if (w < crop_size) else 0
        rem_h = pad_h % 2
        rem_w = pad_w % 2
        pad_dim_h = (pad_h // 2, pad_h // 2 + rem_h)
        pad_dim_w = (pad_w // 2, pad_w // 2 + rem_w)
        # npad is tuple of (n_before, n_after) for each (h,w,d) dimension
        npad = (pad_dim_h, pad_dim_w, (0, 0))
        ndarray = np.pad(ndarray, npad, 'constant', constant_values=0)
        h, w, d = ndarray.shape
    # center crop
    h_offset = (h - crop_size) // 2
    w_offset = (w - crop_size) // 2
    cropped = ndarray[h_offset:(h_offset + crop_size),
              w_offset:(w_offset + crop_size), :]
              
    return cropped

def get_all_images(contour_path, shuffle=True):
    contours = os.listdir(contour_path)
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))

    return contours

def get_all_images_i(contour_path, name, shuffle=True):
    contours = os.listdir(contour_path)
    re_contours = []
    count = 0
    print('---------------------------------------get_all_images_i-------------------------------------------')
    print(len(contours)) #7040
    print(len(name)) #64 16
    for zname in name:
        zstr = '^'+str(zname)+'_'
        #print(zstr)
        for fname in contours:
            if re.search(zstr,fname):
                re_contours.append(fname)
                #print(fname)
                count = count + 1
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(re_contours)
    print('Number of examples: {:d}'.format(len(re_contours)))

    return re_contours

def read_data3(contour,img_path,label_path,edge_path):
    img_full_path = os.path.join(img_path, contour)
    img = cv2.imread(img_full_path, -1).astype('float')

    #img = pre.scale(img) # zscore normalization LYS 4/9/18
    '''
    scaler = MinMaxScaler(feature_range=(0, 1))
    img = scaler.fit_transform(img)

    # 将像素值转换为浮点数类型
    img = img.astype(np.float32)
    # 标准化图像
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    '''

    label_full_path = os.path.join(label_path, contour)
    label = cv2.imread(label_full_path, -1).astype('float')
    edge_full_path = os.path.join(edge_path, contour)
    edge = cv2.imread(edge_full_path, -1).astype('float')

    if img.ndim < 3:
        img = img[..., np.newaxis]
    if label.ndim < 3:
        label = label[..., np.newaxis]
    if edge.ndim < 3:
        edge = edge[..., np.newaxis]

    return img, label, edge

def export_all3_contours(contours, data_path, mask_path, edge_path, crop_size):
    print('Processing {:d} images, labels and edges ...'.format(len(contours)))
    images = np.zeros((len(contours), 256, 256, 1))
    masks = np.zeros((len(contours), 256, 256, 1))
    edges = np.zeros((len(contours), 256, 256, 1))
    for idx, contour in enumerate(contours):
        img, mask, edge = read_data3(contour, data_path, mask_path, edge_path)
        #print(np.max(img))
        #print(np.max(mask))
        img = cv2.resize(center_crop(img, crop_size), (256, 256))
        mask = cv2.resize(center_crop(mask, crop_size), (256, 256))
        edge = cv2.resize(center_crop(edge, crop_size), (256, 256))
        if img.ndim < 3:
            img = img[..., np.newaxis]
        if mask.ndim < 3:
            mask = mask[..., np.newaxis]
        if edge.ndim < 3:
            edge = edge[..., np.newaxis]
        images[idx] = img
        masks[idx] = mask
        edges[idx] = edge

    return images, masks, edges

def export_all4_contours(contours, data_path, mask_path, edge_path, crop_size):
    print('Processing {:d} images, labels and edges ...'.format(len(contours)))
    images = np.zeros((len(contours), 256, 256, 1))
    masks = np.zeros((len(contours), 256, 256, 1))
    masks_orginal = np.zeros((len(contours), 300, 300, 1))
    edges = np.zeros((len(contours), 256, 256, 1))
    for idx, contour in enumerate(contours):
        img, mask, edge = read_data3(contour, data_path, mask_path, edge_path)
        #print(np.max(img))
        #print(np.max(mask))
        img = cv2.resize(center_crop(img, crop_size), (256, 256))
        mask_orginal = cv2.resize(center_crop(mask, crop_size), (300, 300))
        mask = cv2.resize(center_crop(mask, crop_size), (256, 256))
        edge = cv2.resize(center_crop(edge, crop_size), (256, 256))
        if img.ndim < 3:
            img = img[..., np.newaxis]
        if mask.ndim < 3:
            mask = mask[..., np.newaxis]
        if mask_orginal.ndim < 3:
            mask_orginal = mask_orginal[..., np.newaxis]
        if edge.ndim < 3:
            edge = edge[..., np.newaxis]
        images[idx] = img
        masks[idx] = mask
        masks_orginal[idx] = mask_orginal
        edges[idx] = edge

    return images, masks, masks_orginal, edges