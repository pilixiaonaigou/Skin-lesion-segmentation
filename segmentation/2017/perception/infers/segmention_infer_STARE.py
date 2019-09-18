# -*- coding: utf-8 -*-
"""
Copyright (c) 2019. All rights reserved.
Created by Peng Tang on 2019/1/7
"""
import glob,cv2,numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json,load_model
from perception.bases.infer_base import InferBase
from configs.utils.img_utils import get_test_patches,pred_to_patches,recompone_overlap
from configs.utils.utils import visualize,gray2binary
import gc
test_img_path = '..\\Retina-VesselNet-master\\experiments\\VesselNet\\test\\probmap\\'

class SegmentionInfer(InferBase):
	def __init__(self,config):
		super(SegmentionInfer, self).__init__(config)
		self.load_model()

	def load_model(self):
		self.model = model_from_json(open(self.config.hdf5_path+self.config.exp_name + '_architecture.json').read())
#		self.model =load_model(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		print('loading_orig_weights')
#		self.model.load_weights(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		#self.model.load_weights('./keras.model')
	def analyze_name(self,path):
		return (path.split('\\')[-1]).split(".")[0]

	def predict(self):
		predList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
		print(len(predList))
		for index,path in enumerate(predList):
			orgImg_temp=plt.imread(predList[index])
#			orgImg=orgImg_temp[:,:,1]
			thresh = 0.8
			print('G_ratio:{} R_ratio:{}'.format(thresh,(1-thresh)))
			orgImg=orgImg_temp[:,:,1]*thresh+orgImg_temp[:,:,0]*(1-thresh)
			print("[Info] Analyze filename...",self.analyze_name(path))
			height,width=orgImg.shape[:2]
			orgImg = np.reshape(orgImg, (height,width,1))
			patches_pred,new_height,new_width,adjustImg=get_test_patches(orgImg,self.config)
#			self.model.load_weights('./metric/STARE/weights/STARE_1_weight/keras_{}.model'.format(index))
			self.model.load_weights('./metric/STARE/Unetweights/0.9/keras_{}.model'.format(index))

			print("[Info] loading weight....{}".format(index))
			predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
			pred_patches=pred_to_patches(predictions,self.config)

			pred_imgs=recompone_overlap(pred_patches,self.config,new_height,new_width)
			gc.collect()
			pred_imgs=pred_imgs[:,0:height,0:width,:]

			adjustImg=adjustImg[0,0:height,0:width,:]
			print(adjustImg.shape)
			probResult=pred_imgs[0,:,:,0]
			binaryResult=gray2binary(probResult)
			resultMerge=visualize([adjustImg,binaryResult],[1,2])

			resultMerge=cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)

			cv2.imwrite(self.config.test_result_path+self.analyze_name(path)+"_merge.jpg",resultMerge)
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.bmp", (probResult*255).astype(np.uint8))
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + ".tif", (binaryResult*255).astype(np.uint8))
			np.save(self.config.test_result_path + self.analyze_name(path) + ".npy", binaryResult)
			

