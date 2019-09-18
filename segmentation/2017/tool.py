# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 21:14:25 2018

@author: hyc
"""

import os 
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def MetricOnHRF(mask_path,result_path,gt_path,threshold=128):   
    tp = []
    fn = []
    tn = []
    fp = []
    target_path = './metric/HRF/Final_Result/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)    
    for thresh in tqdm(range(127,128)):
        temp_tp = 0
        temp_tn = 0
        temp_fp = 0
        temp_fn = 0
        AUC = 0

        for name in os.listdir(gt_path):
            gt_name = os.path.join(gt_path,name)
            gt =  np.array(Image.open((gt_name)))
            gt = cv2.resize(gt,(1727,1168))
            mask_name = name.replace('.tif','_mask.tif')
            mask_name = os.path.join(mask_path,mask_name)
            mask = cv2.imread(mask_name,0)
            mask = cv2.resize(mask,(1727,1168))
            result_name = name.replace('.tif','_prob.bmp')
#            print(result_name)
            result_name_path = os.path.join(result_path,result_name)
#            print(result_name_path)
            result = cv2.imread(result_name_path,0)
            
            result_2_name = result_2_path + result_name
#            print(result_2_name)
            result_2 = cv2.imread(result_2_name,0)
            
            result_3_name = result_3_path + result_name
            
            result_3 = cv2.imread(result_3_name,0)

            result_4_name = result_4_path + result_name
            
            result_4 = cv2.imread(result_4_name,0)
            
            result_5_name = result_5_path + result_name
            
            result_5 = cv2.imread(result_5_name,0)

#            print(result,result_2,result_3,result_4,result_5)  
#            print()
            result = (result*0.2+result_2*0.2+result_3*0.2+result_4*0.2+result_5*0.2)
#            
#            result[result>=127]=255
#            result[result<127]=0
#            cv2.imwrite(target_path+name,result)              
#            result = (result*0.2+result_2*0.2+result_3*0.2+result_4*0.2+result_5*0.2)
            
            prob_img = result
            gt_img = gt
#            numpy_name = result_name.replace('test_prob.bmp','test.npy')
#            prob_numpy = np.load(numpy_name)
            FOV_result = []
            FOV_gt = []
#            total_result += result
#            total_gt += gt
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i][j] > 0 :
                        FOV_result.append(result[i][j])
                        FOV_gt.append(gt[i][j])
            
            prob_img = np.array(FOV_result)
#            prob_img[prob_img>120]=128
            gt_img = (np.array(FOV_gt)>0 )* 1
#            total_result += result
#            total_gt += gt
#            print(gt_img.shape)
#            print(prob_img.shape)
            AUC+= roc_auc_score(gt_img,prob_img)
#        print(AUC/20)
            new_result = (np.array(prob_img)>thresh)*1
            new_gt = (np.array(gt_img)>0)*1
#        
            temp_tp += np.sum(new_result * new_gt)
            temp_tn += np.sum((1-new_result) * (1-new_gt))
            temp_fn += np.sum((1-new_result) *new_gt)
            temp_fp += np.sum(new_result * (1-new_gt))
#        AUC+= roc_auc_score(total_gt/20,total_result/20)
        print(AUC/30)            
#        print(AUC)
        tp.append(temp_tp)
        tn.append(temp_tn)
        fp.append(temp_fp)
        fn.append(temp_fn)
##    print(tp)
#    
    tp = np.asarray(tp).astype('float32')    
    tn = np.asarray(tn).astype('float32')
    fp = np.asarray(fp).astype('float32')
    fn = np.asarray(fn).astype('float32')
##    print(tp)
    sen =  tp / (tp+fn)
    spe =  tn / (tn+fp)
    acc = (tp+tn)/(tp+fn+fp+tn)
    threshold = np.argmax(acc)
    N = (tp+fn+fp+tn)
    P = (tp+fp)/N
    S = (tp+fn)/N
    MCC = (tp/N - S*P)/np.sqrt(P*S*(1-P)*(1-S))
    print('MCC:{}'.format(MCC))
# 
    print('sen: {} spe: {} acc: {} '.format(sen[threshold],spe[threshold],acc[threshold]))
#MetricOnChasebd(mask_path,result_path,gt_path,threshold=0)
    
def Metric(mask_path,result_path,gt_path,write_path,threshold=128):   
    tp = []
    fn = []
    tn = []
    fp = []
    target_path = './metric/DRIVE/Final_Result/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    if not os.path.exists(write_path):
        os.mkdir(write_path)
#        
#    if not os.path
    for thresh in tqdm(range(120,130)):
        write_sub_path = write_path+'/{}/'.format(thresh)
        if not os.path.exists(write_sub_path):
            os.mkdir(write_sub_path)
            
        temp_tp = 0
        temp_tn = 0
        temp_fp = 0
        temp_fn = 0
        AUC = 0

        for name in os.listdir(gt_path):
            gt_name = os.path.join(gt_path,name)
            gt =  np.array(Image.open((gt_name)))
#            print(name)
            mask_name = name.replace('manual1.tif','test_mask.tif')
#            mask_name = name.replace('manual1.tif','training_mask.tif')

            mask_name = os.path.join(mask_path,mask_name)
#            print(mask_name)
            
            mask = cv2.imread(mask_name,0)
#            print(mask.shape)
            result_name = name.replace('manual1.tif','test_prob.bmp')
#            print(result_name)
#            result_name = name.replace('manual1.tif','training_prob.bmp')

            result_name_path = os.path.join(result_10_path,result_name.replace('bmp','tif'))
            result_10 = cv2.imread(result_name_path,0)
            
            result_11_name = result_11_path + result_name.replace('bmp','tif')
            
            result_11 =cv2.imread(result_11_name,0)
            
            result_2_name = result_2_path + result_name
#            print(result_2_name)
            result_2 = cv2.imread(result_2_name,0)
            
            result_3_name = result_3_path + result_name
            
            result_3 = cv2.imread(result_3_name,0)

            result_4_name = result_4_path + result_name
            
            result_4 = cv2.imread(result_4_name,0)
            
            result_5_name = result_5_path + result_name
            
            result_5 = cv2.imread(result_5_name,0)
            
            result_6_name = result_6_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_6 = cv2.imread(result_6_name,0)    

            result_7_name = result_7_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_7 = cv2.imread(result_7_name,0)   


            result_8_name = result_8_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_8 = cv2.imread(result_8_name,0)   


            result_9_name = result_9_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_9= cv2.imread(result_9_name,0)   

            result_12_name = result_12_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_12= cv2.imread(result_12_name,0)   

            result_13_name = result_13_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_13= cv2.imread(result_13_name,0)   
            
            result_14_name = result_14_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_14= cv2.imread(result_14_name,0)   

            result_15_name = result_15_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_15= cv2.imread(result_15_name,0)  
            
            result_16_name = result_16_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_16= cv2.imread(result_16_name,0)   
            
            result_17_name = result_17_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_17= cv2.imread(result_17_name,0)   

            result_18_name = result_18_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_18= cv2.imread(result_18_name,0)    
            
            result_19_name = result_19_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_19= cv2.imread(result_19_name,0)  

            result_20_name = result_20_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_20= cv2.imread(result_20_name,0)
              
            result_21_name = result_21_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_21= cv2.imread(result_21_name,0)

            result_22_name = result_22_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_22= cv2.imread(result_22_name,0)

            result_23_name = result_23_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_23= cv2.imread(result_23_name,0)

            result_24_name = result_24_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_24= cv2.imread(result_24_name,0)

            result_25_name = result_25_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_25= cv2.imread(result_25_name,0)

            result_26_name = result_26_path + result_name.replace('bmp','tif')
#            result_6_name = result_6_path + result_name
            result_26= cv2.imread(result_26_name,0)

            
#            print(result,result_3,result_4,result_5,result_6)
#            print(result_1_name)            
#            result = (result_2*1.0+result_3*1.0+result_4*1.0+result_5*1.0+result_6*1.0)/5
#            result = (result*0.2+result_2*0.2+result_3*0.2+result_4*0.2+result_5*0.2+result_6*0.2)
#            result = (result_2*1.0+result_3*1.0+result_4*1.0+result_5*1.0+result_6*1.0)/6
#            result = result_2*1.0

#            result = (result_1*1.0+result*1.0+result_2*1.0+result_3*1.0+result_4*1.0+result_5*1.0+result_6*1.0)/7
#            result = (result_1*1.0+result*1.0+result_2*1.0+result_3*1.0+result_4*1.0+result_5*1.0+result_6*1.0+result_7*1.0)/8
#            result = (result_1*1.0+result*1.0+result_2*1.0+result_3*1.0+result_4*1.0+result_5*1.0+result_6*1.0+result_7*1.0)/8
            result = (result_2*1.0+result_3*1.0+result_4*1.0+result_5*1.0+result_6*1.0+result_7*1.0+result_8*1.0+result_9*1.0+result_10*1.0+result_12*1.0+result_13*1.0+result_14*1.0+result_16*1.0+result_17*1.0+result_19*1.0+result_21)/15
#            result = (result_2*1.0+result_3*1.0+result_4*1.0+result_5*1.0+result_6*1.0)/5
#            result = (result_2*1.0+result_3*1.0+result_4*1.0+result_5*1.0+result_6*1.0+result_25*1.0+result_24*1.0+result_23*1.0+result_22*1.0)/9
#            result = (result_7*1.0+result_8*1.0+result_9*1.0+result_10*1.0+result_12*1.0+result_13*1.0+result_14*1.0+result_16*1.0+result_17*1.0+result_19*1.0+result_21)/10
#            result = (result_8*1.0+result_14*1.0+result_17*1.0+result_20*1.0+result_21)/5
#            result = (result_22*1.0+result_23*1.0+result_24*1.0+result_25*1.0+result_26*1.0)/5
#            result = 
#            result = (result_7*1.0+result_8*1.0+result_9*1.0+result_10*1.0+result_10*1.0+result_11*1.0+result_12*1.0+result_13*1.0+result_14*1.0+result_15*1.0+result_16*1.0+result_17*1.0+result_18*1.0+result_19*1.0+result_20*1.0+result_21)/11
            write_name = name.replace('manual1.tif','test.png')
            cv2.imwrite(write_sub_path+write_name,(np.array(result)>thresh)*1)            
#            result = (result*1.0+result_7*1.0+result_1*1.0+result_8*1.0)/4
#            result = (result_10*1.0+result_7*1.0+result_8*1.0+result_11*1.0)/4
#            result = (result_16*1.0+result_17*1.0+result_18*1.0)/3
#            result = (result_19*1.0+result_20*1.0)/2
               
#            result1 =result
#            result1[result1>=127]=255
#            result1[result1<127]=0
#            cv2.imwrite(target_path+name,result1)            
            
            prob_img = result
            gt_img = gt
#            numpy_name = result_name.replace('test_prob.bmp','test.npy')
#            prob_numpy = np.load(numpy_name)
            FOV_result = []
            FOV_gt = []
#            total_result += result
#            total_gt += gt
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i][j] > 0 :
                        FOV_result.append(result[i][j])
                        FOV_gt.append(gt[i][j])
            
            prob_img = np.array(FOV_result)
#            prob_img[prob_img>120]=128
            gt_img = (np.array(FOV_gt)>0 )* 1
#            total_result += result
#            total_gt += gt
            AUC+= roc_auc_score(gt_img,prob_img)
#        print(AUC/20)
            new_result = (np.array(prob_img)>thresh)*1
            new_gt = (np.array(gt_img)>0)*1
#        
            temp_tp += np.sum(new_result * new_gt)
            temp_tn += np.sum((1-new_result) * (1-new_gt))
            temp_fn += np.sum((1-new_result) *new_gt)
            temp_fp += np.sum(new_result * (1-new_gt))
#        AUC+= roc_auc_score(total_gt/20,total_result/20)
        print(AUC/20)            
#        print(AUC)
        tp.append(temp_tp)
        tn.append(temp_tn)
        fp.append(temp_fp)
        fn.append(temp_fn)
##    print(tp)
#    
    tp = np.asarray(tp).astype('float32')    
    tn = np.asarray(tn).astype('float32')
    fp = np.asarray(fp).astype('float32')
    fn = np.asarray(fn).astype('float32')
##    print(tp)
    sen =  tp / (tp+fn)
    spe =  tn / (tn+fp)
    acc = (tp+tn)/(tp+fn+fp+tn)
    dice = 2*tp/(2*tp+fp+fn)
    N = (tp+fn+fp+tn)
    P = (tp+fp)/N
    S = (tp+fn)/N
    MCC = (tp/N - S*P)/np.sqrt(P*S*(1-P)*(1-S))
    #    MCC = ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn*fn))
    threshold = np.argmax(acc)

# 
    print('MCC: {}'.format(MCC))
    print('dice: {}'.format(dice))
    
    print('sen: {} spe: {} acc: {} '.format(sen[threshold],spe[threshold],acc[threshold]))

def MetricOnChasebd(mask_path,result_path,gt_path,threshold=128):   
    tp = []
    fn = []
    tn = []
    fp = []
    target_path = './metric/CHASEBD1/Final_Result/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)    
    for thresh in tqdm(range(125,130)):
        temp_tp = 0
        temp_tn = 0
        temp_fp = 0
        temp_fn = 0
        AUC = 0

        for name in os.listdir(gt_path):
            gt_name = os.path.join(gt_path,name)
            gt =  np.array(Image.open((gt_name)))
            mask_name = name.replace('_1stHO.png','.jpg')
            mask_name = os.path.join(mask_path,mask_name)
            mask = cv2.imread(mask_name,0)
            result_name = name.replace('_1stHO.png','_prob.bmp')
#            print(result_name)
            result_name_path = os.path.join(result_path,result_name)
#            print(result_name_path)
            result = cv2.imread(result_name_path,0)

#            result_1_name = result_1_path +
            
            result_2_name = result_2_path + result_name
#            print(result_2_name)
            result_2 = cv2.imread(result_2_name,0)
            
            result_3_name = result_3_path + result_name
            
            result_3 = cv2.imread(result_3_name,0)

            result_4_name = result_4_path + result_name
            
            result_4 = cv2.imread(result_4_name,0)
            
            result_5_name = result_5_path + result_name
            
            result_5 = cv2.imread(result_5_name,0)

#            print(result,result_2,result_3,result_4,result_5)  
#            print()
            result = (result*0.0+result_2*0.0+result_3*0.0+result_4*0.0+result_5*1.0)
#            
#            result[result>=120]=255
#            result[result<120]=0
#            cv2.imwrite(target_path+name,result)              
#            result = (result*0.2+result_2*0.2+result_3*0.2+result_4*0.2+result_5*0.2)
            
            prob_img = result
            gt_img = gt
#            numpy_name = result_name.replace('test_prob.bmp','test.npy')
#            prob_numpy = np.load(numpy_name)
            FOV_result = []
            FOV_gt = []
#            total_result += result
#            total_gt += gt
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i][j] > 0 :
                        FOV_result.append(result[i][j])
                        FOV_gt.append(gt[i][j])
            
            prob_img = np.array(FOV_result)
#            prob_img[prob_img>120]=128
            gt_img = (np.array(FOV_gt)>0 )* 1
#            total_result += result
#            total_gt += gt
#            print(gt_img.shape)
#            print(prob_img.shape)
            AUC+= roc_auc_score(gt_img,prob_img)
#        print(AUC/20)
            new_result = (np.array(prob_img)>thresh)*1
            new_gt = (np.array(gt_img)>0)*1
#        
            temp_tp += np.sum(new_result * new_gt)
            temp_tn += np.sum((1-new_result) * (1-new_gt))
            temp_fn += np.sum((1-new_result) *new_gt)
            temp_fp += np.sum(new_result * (1-new_gt))
#        AUC+= roc_auc_score(total_gt/20,total_result/20)
        print(AUC/8)            
#        print(AUC)
        tp.append(temp_tp)
        tn.append(temp_tn)
        fp.append(temp_fp)
        fn.append(temp_fn)
##    print(tp)
#    
    tp = np.asarray(tp).astype('float32')    
    tn = np.asarray(tn).astype('float32')
    fp = np.asarray(fp).astype('float32')
    fn = np.asarray(fn).astype('float32')
##    print(tp)
    sen =  tp / (tp+fn)
    spe =  tn / (tn+fp)
    acc = (tp+tn)/(tp+fn+fp+tn)
    threshold = np.argmax(acc)
    N = (tp+fn+fp+tn)    
    P = (tp+fp)/N
    S = (tp+fn)/N
    MCC = (tp/N - S*P)/np.sqrt(P*S*(1-P)*(1-S))  
    print('MCC:{}'.format(MCC))
# 
    print('sen: {} spe: {} acc: {} '.format(sen[threshold],spe[threshold],acc[threshold]))
#MetricOnChasebd(mask_path,result_path,gt_path,threshold=0)
    


def Metric_STARE(mask_path,result_path,gt_path,threshold=128):   
    tp = []
    fn = []
    tn = []
    fp = []
    tp1 = []
    fn1 = []
    tn1 = []
    fp1 = []
    target_path = './metric/STARE/Unet_Final_Result/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
#        
#    if not os.path
    for thresh in tqdm(range(125,130)):
        temp_tp = 0
        temp_tn = 0
        temp_fp = 0
        temp_fn = 0
        temp_tp1 = 0
        temp_tn1 = 0
        temp_fp1 = 0
        temp_fn1 = 0        
#        temp_tp = 0
#        temp_tn = 0
#        temp_fp = 0
#        temp_fn = 0        
        AUC = 0
        AUC_PATH = 0
        counter = 0
        for name in os.listdir(gt_path):
            gt_name = os.path.join(gt_path,name)
            gt =  np.array(Image.open((gt_name)))
#            gt = cv2.resize(gt,(1727,1168))
#            mask_name = name.replace('manual1.tif','test_mask.tif')
            mask_name = gt_name.replace('gt_bmp','img_FOV')
            mask = cv2.imread(mask_name,0)
#            mask = cv2.resize(mask,(1727,1168))
#            print(mask_name)
            result_name = name.replace('.bmp','_prob.bmp')
            result_name_path = os.path.join(result_path,result_name)
            result = cv2.imread(result_name_path,0)
            
            result_2_name = result_2_path + result_name
#            print(result_2_name)
            result_2 = cv2.imread(result_2_name,0)
#            result_2 = np.array(Image.open(result_2_name))
#            print(result.shape)
            result_3_name = result_3_path + result_name
            
            result_3 = cv2.imread(result_3_name,0)

            result_4_name = result_4_path + result_name
            
            result_4 = cv2.imread(result_4_name,0)
#            print()
            result_5_name = result_5_path + result_name
            
            result_5 = cv2.imread(result_5_name,0)
            
#            result_6_name = result_6_path + result_name
            
#            result_6 = cv2.imread(result_6_name,0)    
#            print(result.shape)
#            print(result_2.shape)
#            print(result_3.shape)
#            print(result_4.shape)            
#            result = (result*0.0+result_2*0.0+result_3*0.0+result_4*0.0+result_5*1.0)
            result = (result*0.0+result_2*0.2+result_3*0.2+result_4*0.2+result_5*0.2)
                        
#
#            print(result.shape)
#            result1 =result
#            result[result>=127]=255
#            result[result<127]=0
#            cv2.imwrite(target_path+name,result)            
            
            prob_img = result
            gt_img = gt
#            numpy_name = result_name.replace('test_prob.bmp','test.npy')
#            prob_numpy = np.load(numpy_name)
            FOV_result = []
            FOV_gt = []
#            total_result += result
#            total_gt += gt
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i][j] > 0 :
                        FOV_result.append(result[i][j])
                        FOV_gt.append(gt[i][j])
            
            prob_img = np.array(FOV_result)
#            prob_img[prob_img>120]=128
            gt_img = (np.array(FOV_gt)>0 )* 1
#            total_result += result
#            total_gt += gt
            AUC_single = roc_auc_score(gt_img,prob_img)
            
            counter+=1
            AUC += AUC_single
            new_result = (np.array(prob_img)>thresh)*1
            new_gt = (np.array(gt_img)>0)*1
            if counter > 10:
#                AUC += AUC_single
                AUC_PATH += AUC_single
                temp_tp1 += np.sum(new_result * new_gt)
                temp_tn1 += np.sum((1-new_result) * (1-new_gt))
                temp_fn1 += np.sum((1-new_result) *new_gt)
                temp_fp1 += np.sum(new_result * (1-new_gt))
#                tp1.append(temp_tp)
#                tn1.append(temp_tn)
#                fp1.append(temp_fp)
#                fn1.append(temp_fn)
#            AUC += AUC_single
#            print(mask_name,AUC_single)
#        print(AUC/20)
#            new_result = (np.array(prob_img)>thresh)*1
#            new_gt = (np.array(gt_img)>0)*1
#        
            temp_tp += np.sum(new_result * new_gt)
            temp_tn += np.sum((1-new_result) * (1-new_gt))
            temp_fn += np.sum((1-new_result) *new_gt)
            temp_fp += np.sum(new_result * (1-new_gt))
            
#        AUC+= roc_auc_score(total_gt/20,total_result/20)
        print(AUC/20) 
        print(AUC_PATH/10)           
#        print(AUC)
        if counter > 10:
            tp1.append(temp_tp1)
            tn1.append(temp_tn1)
            fp1.append(temp_fp1)
            fn1.append(temp_fn1)
#        print(len(tp1))
        tp.append(temp_tp)
        tn.append(temp_tn)
        fp.append(temp_fp)
        fn.append(temp_fn)
        
##    print(tp)
#    
    tp = np.asarray(tp).astype('float32')    
    tn = np.asarray(tn).astype('float32')
    fp = np.asarray(fp).astype('float32')
    fn = np.asarray(fn).astype('float32')
    
    tp1 = np.asarray(tp1).astype('float32')    
    tn1 = np.asarray(tn1).astype('float32')
    fp1 = np.asarray(fp1).astype('float32')
    fn1 = np.asarray(fn1).astype('float32')   
##    print(tp)
    sen =  tp / (tp+fn)
    spe =  tn / (tn+fp)
    acc = (tp+tn)/(tp+fn+fp+tn)
    
    sen1 =  tp1 / (tp1+fn1)
    spe1 =  tn1 / (tn1+fp1)
    acc1 = (tp1+tn1)/(tp1+fn1+fp1+tn1)
    N = (tp+fn+fp+tn)    
    P = (tp+fp)/N
    S = (tp+fn)/N
    MCC = (tp/N - S*P)/np.sqrt(P*S*(1-P)*(1-S))    
    threshold = np.argmax(acc)
    threshold1 = np.argmax(acc1)
#    print(len(tp[10:]))
    print('MCC : {}'.format(MCC))
# 
    print('sen: {} spe: {} acc: {} '.format(sen[threshold],spe[threshold],acc[threshold]))
    
    print('path_sen: {} path_spe: {} path)acc: {} '.format(sen1[threshold1],spe1[threshold1],acc1[threshold1]))


    
    
def DemoMetric(mask_path,result_path,gt_path,threshold=128):   
    tp = []
    fn = []
    tn = []
    fp = []
    target_path = './metric/DRIVE/Final_Result/'
    if not os.path.exists(target_path):
        os.mkdir(target_path)
#        
#    if not os.path
    for thresh in tqdm(range(127,128)):
        temp_tp = 0
        temp_tn = 0
        temp_fp = 0
        temp_fn = 0
        AUC = 0

        for name in os.listdir(gt_path):
            gt_name = os.path.join(gt_path,name)
            gt =  np.array(Image.open((gt_name)))
#            mask_name = name.replace('manual1.tif','test_mask.tif')
            mask_name = name.replace('manual1.tif','training_mask.gif')

            mask_name = os.path.join(mask_path,mask_name)
#            print(mask_name.split('/')[-1])
            
#            mask = cv2.imread(mask_name,0)
            mask = np.array(Image.open(mask_name))
#            print(mask.shape)
#            result_name = name.replace('manual1.tif','test_prob.bmp')
            result_name = name.replace('manual1.tif','training_prob.bmp')

            result_name_path = os.path.join(result_path,result_name)
            result = cv2.imread(result_name_path,0)
                     
            
            prob_img = result
            gt_img = gt
            FOV_result = []
            FOV_gt = []
#            total_result += result
#            total_gt += gt
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i][j] > 0 :
                        FOV_result.append(result[i][j])
                        FOV_gt.append(gt[i][j])
            
            prob_img = np.array(FOV_result)
#            prob_img[prob_img>120]=128
            gt_img = (np.array(FOV_gt)>0 )* 1
#            total_result += result
#            total_gt += gt
            auc = roc_auc_score(gt_img,prob_img)
            print('{} AUC : {}'.format(mask_name.split('/')[-1],auc))        
            AUC+= roc_auc_score(gt_img,prob_img)
#        print(AUC/20)
            new_result = (np.array(prob_img)>thresh)*1
            new_gt = (np.array(gt_img)>0)*1
#        
            temp_tp += np.sum(new_result * new_gt)
            temp_tn += np.sum((1-new_result) * (1-new_gt))
            temp_fn += np.sum((1-new_result) *new_gt)
            temp_fp += np.sum(new_result * (1-new_gt))
#        AUC+= roc_auc_score(total_gt/20,total_result/20)
#        print(AUC/20)            
#        print(AUC)
        tp.append(temp_tp)
        tn.append(temp_tn)
        fp.append(temp_fp)
        fn.append(temp_fn)
        print('AUC : {:.4f}'.format(AUC/20))
##    print(tp)
#    
    tp = np.asarray(tp).astype('float32')    
    tn = np.asarray(tn).astype('float32')
    fp = np.asarray(fp).astype('float32')
    fn = np.asarray(fn).astype('float32')
##    print(tp)
    sen =  tp / (tp+fn)
    spe =  tn / (tn+fp)
    acc = (tp+tn)/(tp+fn+fp+tn)
    threshold = np.argmax(acc)

# 
    print('sen: {:.4f} spe: {:.4f} acc: {:.4f} '.format(sen[threshold],spe[threshold],acc[threshold]))
    
    
    
if __name__ == '__main__':
#    mask_path = './metric/HRF/mask/'
#    result_path = './metric/HRF/Unet/0.6/'
#    gt_path = './metric/HRF/gt/'
#    result_2_path = './metric/HRF/Unet/0.7/'
#    result_3_path = './metric/HRF/Unet/0.8/'
#    result_4_path = './metric/HRF/Unet/0.9/'
#    result_5_path = './metric/HRF/Unet/1/'
#    
#    mask_path = './metric/HRF/mask/'
#    result_path = './metric/HRF/TCModelResult/0.6/'
#    gt_path = './metric/HRF/gt/'
#    result_2_path = './metric/HRF/TCModelResult/0.7/'
#    result_3_path = './metric/HRF/TCModelResult/0.8/'
#    result_4_path = './metric/HRF/TCModelResult/0.9/'
#    result_5_path = './metric/HRF/TCModelResult/1/'   
#    
#    
#    MetricOnHRF(mask_path,result_path,gt_path,threshold=128)
#    print(1)
    mask_path = './metric/DRIVE/mask/'
#    result_path = './metric/CROSS_TRAING/DRIVE/CHASE_DB1/0.6/'
#    gt_path = './metric/DRIVE/groundtruth/'
#    result_2_path = './metric/CROSS_TRAING/DRIVE/CHASE_DB1/0.7/'
#    result_3_path = './metric/CROSS_TRAING/DRIVE/CHASE_DB1/0.8/'
#    result_4_path = './metric/CROSS_TRAING/DRIVE/CHASE_DB1/0.9/'
#    result_5_path = './metric/CROSS_TRAING/DRIVE/CHASE_DB1/1.0/'
#    result_6_path = './metric/CROSS_TRAING/DRIVE/CHASE_DB1/0.9/'
#
#    result_path = './metric/DRIVE/deepattention/result_0.6G/'
#    gt_path = './metric/DRIVE/groundtruth/'
#    result_2_path = './metric/DRIVE/deepattention/result_0.7G/'
#    result_3_path = './metric/DRIVE/deepattention/result_0.8G/'
#    result_4_path = './metric/DRIVE/deepattention/result_0.9G/'
#    result_5_path = './metric/DRIVE/deepattention/result_G/'
#    result_6_path = './metric/DRIVE/deepattention/result_G/'
#   
#    gt_path = './metric/DRIVE/groundtruth/'
#    result_path = './metric/DRIVE/uNET/result_0.6G/' 
##    gt_path = './metric/DRIVE/groundtruth/'
#    result_2_path = './metric/DRIVE/uNET/result_0.6G/'
#    result_3_path = './metric/DRIVE/uNET/result_0.7G/'
#    result_4_path = './metric/DRIVE/uNET/result_0.8G/'
#    result_5_path = './metric/DRIVE/uNET/result_0.9G/'
#    result_6_path = './metric/DRIVE/uNET/result_G/'





    
    gt_path = './metric/DRIVE/groundtruth/'
    result_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.6G_0.3R_0.1B/' 
#    gt_path = './metric/DRIVE/groundtruth/'
    result_1_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.6G_0.4R_0.0B/'
    result_2_path = './metric/DRIVE/result_DRRUnet_0.6G_1/'
    result_3_path = './metric/DRIVE/result_DRRUnet_0.8G_3/'
    result_4_path = './metric/DRIVE/result_DRRuNET_0.7G/'
    result_5_path = './metric/DRIVE/result_DRRUnet_G_2/'
    result_6_path = './metric/DRIVE/result/'    
    
    result_10_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.6G_0.3R_0.1B/' 
    result_11_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.6G_0.4R_0.0B/'    
    result_7_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.6G_0.2R_0.2B/'    
    result_8_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.6G_0.0R_0.4B/'
    result_9_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.6G_0.1R_0.3B/'


    result_12_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.7G_0.2R_0.1B/' 
    result_13_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.7G_0.1R_0.2B/'    
    result_14_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.7G_0.0R_0.3B/'    
    result_15_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.7G_0.3R_0.0B/'
#    result_9_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.6G_0.1R_0.3B/'
    result_16_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.8G_0.1R_0.1B/'    
    result_17_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.8G_0.0R_0.2B/'    
    result_18_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.8G_0.2R_0.0B/'
#
    result_19_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.9G_0.0R_0.1B/'    
    result_20_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.9G_0.1R_0.0B/'
#
    result_21_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_1.0G_0.0R_0.0B/'

    result_22_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.95G_0.05R_0.0B/'
    result_23_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.85G_0.15R_0.0B/'
    result_24_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.75G_0.25R_0.0B/'
    result_25_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_128_0.65G_0.35R_0.0B/'
    result_26_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_32_0.6G_0.4R_0.0B/'

    
#    result_22_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_32_1.0G_0.0R_0.0B/'
#    result_23_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_32_0.9G_0.1R_0.0B/'
#    result_24_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_32_0.8G_0.2R_0.0B/'
#    result_25_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_32_0.7G_0.3R_0.0B/'
#    result_26_path = './experiments/VesselNet/dataset/DRIVE/validate/result_demo_32_0.6G_0.4R_0.0B/'


    
#    result_path = './metric/DRIVE/Difference_score/result/' 
#    gt_path = './metric/DRIVE/Difference_score/img_gt/'
#    result_2_path = './metric/DRIVE/Difference_score/result/'
#    result_3_path = './metric/DRIVE/Difference_score/result/'
#    result_4_path = './metric/DRIVE/Difference_score/result/'
#    result_5_path = './metric/DRIVE/Difference_score/result/'
#    result_6_path = './metric/DRIVE/Difference_score/result/'
#    DemoMetric(mask_path,result_path,gt_path,threshold=0)   
    write_path = './15_ensemble_submit_result/'
    Metric(mask_path,result_path,gt_path,write_path,threshold=0)    
#    #CHASE_BD1
#    mask_path = './metric/CHASEBD1/FOV/'
#    result_path = './metric/CHASEBD1/our_designed_sub_model_1/result_0.6/' 
#    gt_path = './metric/CHASEBD1/groundtruth/'
#    result_2_path = './metric/CHASEBD1/our_designed_sub_model_1/result_0.7/'
#    result_3_path = './metric/CHASEBD1/our_designed_sub_model_1/result_0.8/'
#    result_4_path = './metric/CHASEBD1/our_designed_sub_model_1/result_0.9_2/'
#    result_5_path = './metric/CHASEBD1/our_designed_sub_model_1/result_1/'
#
#    mask_path = './metric/CHASEBD1/FOV/'
#    result_path = './metric/CHASEBD1/unet/result_0.6g/' 
#    gt_path = './metric/CHASEBD1/groundtruth/'
#    result_2_path = './metric/CHASEBD1/unet/result_0.7G/'
#    result_3_path = './metric/CHASEBD1/unet/result_0.8G/'
#    result_4_path = './metric/CHASEBD1/unet/result_0.9G/'
#    result_5_path = './metric/CHASEBD1/unet/result_G/'
#
#
#
#    MetricOnChasebd(mask_path,result_path,gt_path,threshold=0)

#    result_path = './metric/CHASEBD1/our_designed_sub_model_1/result/' 
#    gt_path = './metric/CHASEBD1/groundtruth/'
#    result_2_path = './metric/CROSS_TRAING/CHASE_DB1/DRIVE/0.7/'
#    result_3_path = './metric/CROSS_TRAING/CHASE_DB1/DRIVE/0.8/'
#    result_4_path = './metric/CROSS_TRAING/CHASE_DB1/DRIVE/0.8/'
#    result_5_path = './metric/CROSS_TRAING/CHASE_DB1/DRIVE/1/'
#    
#    
#    result_path = './metric/CHASEBD1/unet/result_0.6g/' 
#    gt_path = './metric/CHASEBD1/groundtruth/'
#    result_2_path = './metric/CHASEBD1/unet/result_0.7G/' 
#    result_3_path = './metric/CHASEBD1/unet/result_0.8G/' 
#    result_4_path = './metric/CHASEBD1/unet/result_0.9G/' 
#    result_5_path = './metric/CHASEBD1/unet/result_G/'    

#    result_path = './metric/CHASEBD1/unet/result_0.6g/' 
#    gt_path = './metric/CHASEBD1/groundtruth/'
#    result_2_path = './metric/CHASEBD1/unet/result_0.7G/' 
#    result_3_path = './metric/CHASEBD1/unet/result_0.8G/' 
#    result_4_path = './metric/CHASEBD1/unet/result_0.9G/' 
#    result_5_path = './metric/CHASEBD1/unet/result_G/' 

    
    #STARE
#    mask_path = './metric/STARE/img_FOV_1/'
#    result_path = './metric/STARE/Unet/result_0.6/' 
#    gt_path = './metric/STARE/gt_bmp/'
#    result_2_path = './metric/STARE/Unet/result_0.7/'
#    result_3_path = './metric/STARE/Unet/result_0.8/'
#    result_4_path = './metric/STARE/Unet/result_0.9/'
#    result_5_path = './metric/STARE/Unet/result_1.0/'
#    
#    mask_path = './metric/STARE/img_FOV_1/'
#    result_path = './metric/STARE/result_0.6/' 
#    gt_path = './metric/STARE/gt_bmp/'
#    result_2_path = './metric/STARE/result_0.7/'
#    result_3_path = './metric/STARE/result_0.8_2/'
#    result_4_path = './metric/STARE/result_0.9/'
#    result_5_path = './metric/STARE/result_G_1/'    
#    Metric_STARE(mask_path,result_path,gt_path,threshold=0)
#    
#    mask_path = './metric/STARE/img_FOV_1/'
#    result_path = './metric/CROSS_TRAING/STARE/CHASE_DB1/0.7/' 
#    gt_path = './metric/STARE/gt_bmp/'
#    result_2_path = './metric/CROSS_TRAING/STARE/CHASE_DB1/0.8/'
#    result_3_path = './metric/CROSS_TRAING/STARE/CHASE_DB1/0.9/'
#    result_4_path = './metric/CROSS_TRAING/STARE/CHASE_DB1/0.6/'
#    result_5_path = './metric/CROSS_TRAING/STARE/CHASE_DB1/1/'
        
#    result_path = './metric/CROSS_TRAING/STARE/DRIVE/0.6/' 
#    gt_path = './metric/STARE/gt_bmp/'
#    result_2_path = './metric/CROSS_TRAING/STARE/DRIVE/0.8/'
#    result_3_path = './metric/CROSS_TRAING/STARE/DRIVE/0.9/'
#    result_4_path = './metric/CROSS_TRAING/STARE/DRIVE/0.7/'
#    result_5_path = './metric/CROSS_TRAING/STARE/DRIVE/1/'
       
    
    ##################################################
#    Metric(mask_path,result_path,gt_path,threshold=0)   
    
#    Metric_STARE(mask_path,result_path,gt_path,threshold=0)
#    MetricOnChasebd(mask_path,result_path,gt_path,threshold=0)
#    ##################################
    
    
    
#def GetFoVDRIVE(img_path,gt_path,FOV_img_path,FOV_gt_path,dilameter=270,Dataset='DRIVE',mode='training'):
#    
#    if not os.path.exists(FOV_img_path):
#        os.mkdir(FOV_img_path)
#    if not os.path.exists(FOV_gt_path):
#        os.mkdir(FOV_gt_path)
#        
#    for i in tqdm(os.listdir(img_path)):
#        img_name =   i
#        if Dataset == 'DRIVE':
#            if mode == 'training':
#                mask_name = i.replace('training','manual1')
#            elif mode =='test':
#                mask_name =  i.replace('test','manual1')
#        
#        img = cv2.imread(img_path+img_name)
#        gt = np.array(Image.open(gt_path+mask_name))
#        
#        center_h = img.shape[0] // 2
#        center_w = img.shape[1] // 2
#        
#        x1 = center_h - dilameter
#        x2 = center_h + dilameter
#        y1 = center_w - dilameter
#        y2 = center_w + dilameter
#        
#        img_FOV = img[x1:x2,y1:y2]
#        gt_FOV = gt[x1:x2,y1:y2]
#        
#        cv2.imwrite(FOV_img_path+img_name,img_FOV,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
#        cv2.imwrite(FOV_gt_path+mask_name,gt_FOV,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
#        
