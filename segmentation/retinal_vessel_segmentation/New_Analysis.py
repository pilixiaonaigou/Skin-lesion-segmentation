# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 21:22:03 2019

@author: hyc
"""
import os 
import cv2
from tqdm import tqdm
import numpy as np
global Dark_list
from PIL import Image
import matplotlib.pyplot as plt
##########train_img_path##############################################
#img_path = './experiments/VesselNet/dataset/DRIVE/train/origin/'
#gt_path = './experiments/VesselNet/dataset/DRIVE/train/gt/'
#bright_path = './experiments/VesselNet/dataset/DRIVE/train/bright/'
#dark_path = './experiments/VesselNet/dataset/DRIVE/train/dark/'
######################################################################


#########test_img_path##############################################
#img_path = './experiments/VesselNet/dataset/DRIVE/validate/origin/'
#gt_path = './experiments/VesselNet/dataset/DRIVE/validate/groundtruth/'
#bright_path = './experiments/VesselNet/dataset/DRIVE/validate/bright/'
#dark_path = './experiments/VesselNet/dataset/DRIVE/validate/dark/'
#####################################################################



def GrayLevelCal(img_path,gt_path,bright_path,dark_path):
    
    bright_gt_path = bright_path.replace('bright','bright_gt')
    dark_gt_path = dark_path.replace('dark','dark_gt')
    
    for path in [bright_path,bright_gt_path,dark_gt_path,dark_path]:
        if not os.path.exists(path):
            os.mkdir(path)

#    if not os.path.exists(dark_path):
#        os.mkdir(bright_path)
#        

    name_list = []
    ratio_list = []
    Bright_list = []
#    Bright_gt_list = []
#    Dark_list = []
    for i in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path,i),0)
#        gt_name = i.replace('training','manual1')
#        gt  = cv2.imread(os.path.join(gt_path,gt_name),0)
        w,h = img.shape
        ratio = np.sum(img)/(w*h)
        print("{} Gray_Ratio:{}".format(i,ratio))
        name_list.append(i)
        ratio_list.append(ratio)
    for  j in range(20):
        index = np.argmax(ratio_list)
        Bright_list.append(name_list[index])
#        Bright_gt_list.append(gt_name)
        ratio_list.pop(index)
        name_list.pop(index)
    
    Dark_list = name_list
    print('亮度为高的列表：{}'.format(Bright_list))
    print('亮度为低的列表：{}'.format(Dark_list))
    
    for img_name in tqdm(Bright_list):
#        print(img_name)
        img = cv2.imread(os.path.join(img_path,img_name))
        cv2.imwrite(bright_path+img_name,img)
        gt_name = img_name.replace('training','manual1')
#        gt_name = img_name.replace('test','manual1')
#        print(gt_path+gt_name)
        gt = np.array(Image.open(gt_path+gt_name))
#        print(gt.shape)
        cv2.imwrite(bright_gt_path+gt_name,gt)
        
    for img_name in tqdm(Dark_list):
        img = cv2.imread(os.path.join(img_path,img_name))
        cv2.imwrite(dark_path+img_name,img)
        gt_name = img_name.replace('training','manual1')
#        gt_name = img_name.replace('test','manual1')
        gt = np.array(Image.open(gt_path+gt_name))
        print(dark_gt_path+gt_name)
        cv2.imwrite(dark_gt_path+gt_name,gt)

def clahe_equalized(imgs):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    imgs_equalized = clahe.apply(np.array(imgs, dtype = np.uint8))
    
    return imgs_equalized


def mask_dir_if_not(path):
    if not os.path.exists(path):
        os.mkdir(path)


green = [0,255,0]
red = [0,0,255]
black = [0,0,0]
white = [255,255,255]
gray = [128,128,128]
color_list = [green,red,black,white]

#######################################
#gt_path = './experiments/VesselNet/dataset/DRIVE/train/gt/'
#open_operation_path = './experiments/VesselNet/dataset/DRIVE/train/open_operation_path/'
#dalition_path = './experiments/VesselNet/dataset/DRIVE/train/dalition/'
#thick_path = './experiments/VesselNet/dataset/DRIVE/train/thick/'
#thick__bordor_path = './experiments/VesselNet/dataset/DRIVE/train/thick_border/'
#final_mask_path = './experiments/VesselNet/dataset/DRIVE/train/final_mask/'
#final_gt_path = './experiments/VesselNet/dataset/DRIVE/train/final_gt/'
######################################

#######################################
gt_path = './experiments/VesselNet/dataset/DRIVE/validate/groundtruth/'
open_operation_path = './experiments/VesselNet/dataset/DRIVE/validate/open_operation_path/'
dalition_path = './experiments/VesselNet/dataset/DRIVE/validate/dalition/'
thick_path = './experiments/VesselNet/dataset/DRIVE/validate/thick/'
thick__bordor_path = './experiments/VesselNet/dataset/DRIVE/validate/thick_border/'
final_mask_path = './experiments/VesselNet/dataset/DRIVE/validate/final_mask/'
final_gt_path = './experiments/VesselNet/dataset/DRIVE/validate/final_gt/'
######################################





mask_dir_if_not(open_operation_path)
mask_dir_if_not(dalition_path)
mask_dir_if_not(thick_path)
mask_dir_if_not(final_mask_path)
mask_dir_if_not(thick__bordor_path)
mask_dir_if_not(final_gt_path)




name = os.listdir(gt_path)[0]
for name in tqdm(os.listdir(gt_path)[:20]):
    
    img = Image.open(gt_path+name)
    img = np.array(img)
    
    #kernel = np.array([[1,1,1],
    #                  [1,1,1],
    #                  [1,1,1]])
    
    kernel_erosion = np.ones((3,3),np.uint8)
    kernel_dalition = np.ones((3,3),np.uint8)
    kernel_dalition1 = np.ones((3,3),np.uint8)
    
    erosion = cv2.erode(img,kernel_erosion,iterations = 1)    
    opening = cv2.dilate(erosion,kernel_dalition,iterations = 1)
    
    thick = img-opening
    cv2.imwrite(thick__bordor_path +name,thick)
    thick_dalition = cv2.dilate(thick,kernel_dalition1,iterations = 1)
    thick_border = thick_dalition-thick
    
    thick[thick==255] = 128
    thick = cv2.cvtColor(thick,cv2.COLOR_GRAY2BGR)
    thick[:,:,2] += thick_border
    #thick = cv2.cvtColor(thick,cv2.GRAY2BGR)
    
    opening_dalition = cv2.dilate(opening,kernel_dalition1,iterations = 1)
    opening_border = opening_dalition - opening
    opening = cv2.cvtColor(opening,cv2.COLOR_GRAY2BGR)
    opening[:,:,1] += opening_border
    
    final = np.zeros(opening.shape,np.uint8)

    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            if np.mean(opening[i][j])==0:
                final[i][j] = thick[i][j]
            else:
                final[i][j] = opening[i][j]
                
    gt = np.zeros(opening.shape[:2],np.uint8)                

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            mask = final[i][j]
            B = mask[0]
            G = mask[1]
            R = mask[2]
            if  B==128 and G==128 and R==128:
                gt[i][j] = 1
            elif  B==0 and G==255 and R==0:
                gt[i][j] = 2   
            elif  B==0 and G==0 and R==255:
                gt[i][j] = 3   
            elif  B==255 and G==255 and R==255:
                gt[i][j] = 4                    

            
    #opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
#    final = opening+thick
#    for i in range(final.shape[0]):
#        for j in range(final.shape[1]):
#            if (final[i][j][1]==255) and (final[i][j][2]==255) and (final[i][j][0]==0):
#                final[i][j] = [0,0,255]
#            if [final[i][j][0],final[i][j][1],final[i][j][2]] not in color_list:
#                final[i][j] = [0,0,0]
                
    cv2.imwrite(thick_path+name,thick)
    cv2.imwrite(open_operation_path+name,opening)
    cv2.imwrite(final_mask_path +name,final)
    cv2.imwrite(final_gt_path +name,gt)    
#    cv2.imwrite(thick__bordor_path +name,thick_border)
#cv2.imwrite(dalition_path+i,dalition)

#cv2.imwrite()
#cv2.imshow('src',img)
#cv2.imshow('show',opening)
#cv2.waitKey(0)


#plt.imshow(img)
#plt.imshow(img)
#img = cv2.imread('./experiments/VesselNet/dataset/DRIVE/train/origin/')        

#GrayLevelCal(img_path,gt_path,bright_path,dark_path)
#def GrayLevelDiff(img_path)
