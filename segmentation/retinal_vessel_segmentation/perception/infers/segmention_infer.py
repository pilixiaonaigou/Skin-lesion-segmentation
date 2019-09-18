# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by Peng Tang on 2019/1/7
"""
import glob,cv2,numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json,load_model
from perception.bases.infer_base import InferBase
from configs.utils.img_utils import get_test_patches,pred_to_patches,recompone_overlap,get_test_patches1
from configs.utils.utils import visualize,gray2binary
import gc
green = [0,255,0]
red = [0,0,255]
black = [0,0,0]
white = [255,255,255]
gray = [128,128,128]

color_list=[black,gray,green,red,white]

test_img_path = '..\\Retina-VesselNet-master\\experiments\\VesselNet\\test\\probmap\\'

class SegmentionInferMulticlass(InferBase):
	def __init__(self,config):
		super(SegmentionInferMulticlass, self).__init__(config)
		self.load_model()

	def load_model(self):
		self.model = model_from_json(open(self.config.hdf5_path+self.config.exp_name + '_architecture.json').read())
#		self.model =load_model(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		print('loading_orig_weights')
		path = './keras_DRIVE_Binary.model'
		print('Weight Path : {}'.format(path))
#		self.model.load_weights(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		self.model = load_model(path)
#		self.model.load_weights('./keras_DRIVE.model')
#		self.model.load_weights('./DRIVE.model')
#        self.G = self.config.G
#        self.R = self.config.R
#        self.B = self.config.B
        
	def analyze_name(self,path):
		return (path.split('\\')[-1]).split(".")[0]

	def predict(self):
		predList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
		print(len(predList))
		for path in predList:
			orgImg_temp=plt.imread(path)
			#orgImg_temp=cv2.resize(orgImg_temp,(1727,1168))
#			orgImg=orgImg_temp[:,:,1]
			thresh = 0.6
			print('G_ratio:{} R_ratio:{}'.format(thresh,(1-thresh)))
			orgImg=orgImg_temp[:,:,1]*thresh+orgImg_temp[:,:,0]*(1-thresh)
			print("[Info] Analyze filename...",self.analyze_name(path))
			height,width=orgImg.shape[:2]
			orgImg = np.reshape(orgImg, (height,width,1))
			patches_pred,new_height,new_width,adjustImg=get_test_patches(orgImg,self.config)
			
			predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
#			print(predictions.shape)
			pred_patches=pred_to_patches(predictions,self.config)

			pred_imgs=recompone_overlap(pred_patches,self.config,new_height,new_width)
#			print(pred_imgs.shape)
			gc.collect()
			pred_imgs=pred_imgs[:,0:height,0:width,:]
			new_construct = np.zeros((pred_imgs.shape[1],pred_imgs.shape[2],3))
			for i in range(pred_imgs.shape[1]):
			    for j in range(pred_imgs.shape[2]):
#			        print(pred_imgs[0][i][j])
			        index = np.argmax(pred_imgs[0][i][j])
#			        print(index)
			        new_construct[i][j] = color_list[index]
                    
                    
#			print(pred_imgs.shape)
			adjustImg=adjustImg[0,0:height,0:width,:]
#			print(adjustImg.shape)
            
			probResult=pred_imgs[0,:,:,0] + pred_imgs[0,:,:,2] + pred_imgs[0,:,:,3]
            
			binaryResult=gray2binary(probResult,0.35)
			resultMerge=visualize([adjustImg,binaryResult],[1,2])

			resultMerge=cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)

			#cv2.imwrite(self.config.test_result_path+self.analyze_name(path)+"_merge.jpg",resultMerge)
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.tif", ((1-probResult)*255).astype(np.uint8))
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_color_mask.tif", (new_construct).astype(np.uint8))

			#cv2.imwrite(self.config.test_result_path + self.analyze_name(path).replace('training','manual1') + ".tif", (binaryResult*255).astype(np.uint8))
			#np.save(self.config.test_result_path + self.analyze_name(path) + ".npy", binaryResult)







class SegmentionInferbinary(InferBase):
	def __init__(self,config):
		super(SegmentionInferbinary, self).__init__(config)
		self.load_model()
		self.G = self.config.G
		self.R = self.config.R
		self.B = self.config.B
		self.color_space = self.config.color_space
		self.multi_proportion = self.config.multi_proportion
        
	def load_model(self):
		self.model = model_from_json(open(self.config.hdf5_path+self.config.exp_name + '_architecture.json').read())
#		self.model =load_model(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		print('loading_orig_weights')
#		self.model.load_weights(self.config.hdf5_path+'/VesselNet_best_weights.h5')
#		self.model = load_model('./keras_DRIVE.model')
		self.model.load_weights('./keras_DRIVE_Binary.model')
#		self.model.load_weights('./DRIVE.model')
	def analyze_name(self,path):
		return (path.split('\\')[-1]).split(".")[0]

	def predict(self):
		predList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
		print(len(predList))
		for path in predList:
			orgImg_temp=plt.imread(path)
			#orgImg_temp=cv2.resize(orgImg_temp,(1727,1168))
#			orgImg=orgImg_temp[:,:,1]
			if self.color_space == 'RGB':
			    print('Color Space: RGB')                
			    if self.multi_proportion : 
        			print('G_ratio:{} R_ratio:{} B_ratio:{}'.format(self.G,self.R,self.B))            
        			orgImg=orgImg_temp[:,:,1]*self.G+orgImg_temp[:,:,0]*self.R+orgImg_temp[:,:,2]*self.B
			    else:
    				orgImg=np.asarray(orgImg_temp)
			elif self.color_space == 'HSV':
			    print('Color Space: HSV')                 
			    orgImg = cv2.cvtColor(orgImg_temp,cv2.COLOR_RGB2HSV)                
              

			elif self.color_space == 'LAB':
			    print('Color Space: LAB')                 
			    orgImg = cv2.cvtColor(orgImg_temp,cv2.COLOR_RGB2LAB)                
   
                
#			print('G_ratio:{} R_ratio:{} B_ratio:{}'.format(self.G,self.R,self.B))            
#			orgImg=orgImg_temp[:,:,1]*self.G+orgImg_temp[:,:,0]*self.R+orgImg_temp[:,:,2]*self.B
			
			print("[Info] Analyze filename...",self.analyze_name(path))
			height,width=orgImg.shape[:2]
			if self.multi_proportion :
			    orgImg = np.reshape(orgImg, (height,width,1))
			else:
			    orgImg = np.reshape(orgImg, (height,width,3))                
            
			patches_pred,new_height,new_width,adjustImg=get_test_patches(orgImg,self.config)
			
			predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
			#pred_patches=pred_to_patches(predictions,self.config)

#			predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
#			pred_patches=pred_to_patches(predictions,self.config)

			pred_imgs=recompone_overlap(predictions,self.config,new_height,new_width)
#			print(pred.shape)            
			gc.collect()
            
			pred_imgs=pred_imgs[:,0:height,0:width,:]

			adjustImg=adjustImg[0,0:height,0:width,:]
#			print(adjustImg.shape)
#            new_
			probResult=pred_imgs[0,:,:,0]
			binaryResult=gray2binary(probResult,0.35)			#cv2.imwrite(self.config.test_result_path+self.analyze_name(path)+"_merge.jpg",resultMerge)
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.tif", (probResult*255).astype(np.uint8))
			#cv2.imwrite(self.config.test_result_path + self.analyze_name(path).replace('training','manual1') + ".tif", (binaryResult*255).astype(np.uint8))
			#np.save(self.config.test_result_path + self.analyze_name(path) + ".npy", binaryResult)



class SegmentionInfer(InferBase):
	def __init__(self,config):
		super(SegmentionInfer, self).__init__(config)
		self.load_model()

	def load_model(self):
		self.model = model_from_json(open(self.config.hdf5_path+self.config.exp_name + '_architecture.json').read())
#		self.model =load_model(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		print('loading_orig_weights')
#		self.model.load_weights(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		self.model = load_model('./keras_DRIVE.model')
#		self.model.load_weights('./keras_DRIVE.model')
#		self.model.load_weights('./DRIVE.model')
	def analyze_name(self,path):
		return (path.split('\\')[-1]).split(".")[0]

	def predict(self):
		predList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
		print(len(predList))
		for path in predList:
			orgImg_temp=plt.imread(path)
			#orgImg_temp=cv2.resize(orgImg_temp,(1727,1168))
#			orgImg=orgImg_temp[:,:,1]
			thresh = 1
			print('G_ratio:{} R_ratio:{}'.format(thresh,(1-thresh)))
			orgImg=orgImg_temp[:,:,1]*thresh+orgImg_temp[:,:,0]*(1-thresh)
			print("[Info] Analyze filename...",self.analyze_name(path))
			height,width=orgImg.shape[:2]
			orgImg = np.reshape(orgImg, (height,width,1))
			patches_pred,new_height,new_width,adjustImg=get_test_patches(orgImg,self.config)
			
			predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
			pred_patches=pred_to_patches(predictions,self.config)

			pred_imgs=recompone_overlap(pred_patches,self.config,new_height,new_width)
			gc.collect()
			pred_imgs=pred_imgs[:,0:height,0:width,:]

			adjustImg=adjustImg[0,0:height,0:width,:]
			print(adjustImg.shape)
			probResult=pred_imgs[0,:,:,0]
			binaryResult=gray2binary(probResult,0.35)
			resultMerge=visualize([adjustImg,binaryResult],[1,2])

			resultMerge=cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)

			#cv2.imwrite(self.config.test_result_path+self.analyze_name(path)+"_merge.jpg",resultMerge)
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.tif", (probResult*255).astype(np.uint8))
			#cv2.imwrite(self.config.test_result_path + self.analyze_name(path).replace('training','manual1') + ".tif", (binaryResult*255).astype(np.uint8))
			#np.save(self.config.test_result_path + self.analyze_name(path) + ".npy", binaryResult)
			

class SegmentionInferprob(InferBase):
	def __init__(self,config):
		super(SegmentionInferprob, self).__init__(config)
		self.load_model()

	def load_model(self):
		self.model = model_from_json(open(self.config.hdf5_path+self.config.exp_name + '_architecture.json').read())
#		self.model =load_model(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		print('loading_orig_weights')
#		self.model.load_weights(self.config.hdf5_path+'/VesselNet_best_weights.h5')
#		self.model = load_model('./keras_CHASE_DB1.model')
		self.model.load_weights('./keras_prob.model')
#		self.model.load_weights('./DRIVE.model')
	def analyze_name(self,path):
		return (path.split('\\')[-1]).split(".")[0]

	def predict(self):
		predList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
#		probList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
#        print(len(predList))
		for path in predList:
			orgImg_temp=plt.imread(path)
#			print(path)
			prob_path = self.config.test_prob_path + path.split('\\')[-1].replace('test','test_prob')
			probImg = plt.imread(prob_path,0)

			#orgImg_temp=cv2.resize(orgImg_temp,(1727,1168))
#			orgImg=orgImg_temp[:,:,1]
			thresh = 1
			print('G_ratio:{} R_ratio:{}'.format(thresh,(1-thresh)))
			orgImg=orgImg_temp[:,:,1]*thresh+orgImg_temp[:,:,0]*(1-thresh)
			print("[Info] Analyze filename...",self.analyze_name(path))
			height,width=orgImg.shape[:2]
			orgImg = np.reshape(orgImg, (height,width,1))
			probImg = np.reshape(probImg, (height,width,1))
			probImg = np.transpose(probImg,(2,0,1))
			
			patches_pred,new_height,new_width,adjustImg=get_test_patches1(orgImg,probImg,self.config)
			
			predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
			pred_patches=pred_to_patches(predictions,self.config)

			pred_imgs=recompone_overlap(pred_patches,self.config,new_height,new_width)
			gc.collect()
			pred_imgs=pred_imgs[:,0:height,0:width,:]

			adjustImg=adjustImg[0,0:height,0:width,:]
			print(adjustImg.shape)
			probResult=pred_imgs[0,:,:,0]
			binaryResult=gray2binary(probResult,0.35)
#			resultMerge=visualize([adjustImg,binaryResult],[1,2])

#			resultMerge=cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)

#			cv2.imwrite(self.config.test_result_path+self.analyze_name(path)+"_merge.jpg",resultMerge)
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.tif", (probResult*255).astype(np.uint8))
			#cv2.imwrite(self.config.test_result_path + self.analyze_name(path).replace('training','manual1') + ".tif", (binaryResult*255).astype(np.uint8))
			#np.save(self.config.test_result_path + self.analyze_name(path) + ".npy", binaryResult)
			
class SegmentionInferbinaryprob(InferBase):
	def __init__(self,config):
		super(SegmentionInferbinaryprob, self).__init__(config)
		self.load_model()

	def load_model(self):
		self.model = model_from_json(open(self.config.hdf5_path+self.config.exp_name + '_architecture.json').read())
#		self.model =load_model(self.config.hdf5_path+'/VesselNet_best_weights.h5')
		print('loading_orig_weights')
#		self.model.load_weights(self.config.hdf5_path+'/VesselNet_best_weights.h5')
#		self.model = load_model('./keras_CHASE_DB1.model')
		self.model.load_weights('./keras_DRIVE.model')
#		self.model.load_weights('./DRIVE.model')
	def analyze_name(self,path):
		return (path.split('\\')[-1]).split(".")[0]

	def predict(self):
		predList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
#		probList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
#        print(len(predList))
		for path in predList:
			orgImg_temp=plt.imread(path)
#			print(path)
			prob_path = self.config.test_prob_path + path.split('\\')[-1].replace('test','test_prob')
			probImg = plt.imread(prob_path,0)

			#orgImg_temp=cv2.resize(orgImg_temp,(1727,1168))
#			orgImg=orgImg_temp[:,:,1]
			thresh = 0.6
			print('G_ratio:{} R_ratio:{}'.format(thresh,(1-thresh)))
			orgImg=orgImg_temp[:,:,1]*thresh+orgImg_temp[:,:,0]*(1-thresh)
			print("[Info] Analyze filename...",self.analyze_name(path))
			height,width=orgImg.shape[:2]
			orgImg = np.reshape(orgImg, (height,width,1))
			probImg = np.reshape(probImg, (height,width,1))
			probImg = np.transpose(probImg,(2,0,1))
			
			patches_pred,new_height,new_width,adjustImg=get_test_patches1(orgImg,probImg,self.config)
			
			predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
#			pred_patches=pred_to_patches(predictions,self.config)

			pred_imgs=recompone_overlap(predictions,self.config,new_height,new_width)
			print(pred.shape)            
			gc.collect()
			pred_imgs=pred_imgs[:,0:height,0:width,:]

			adjustImg=adjustImg[0,0:height,0:width,:]
			print(adjustImg.shape)
			probResult=pred_imgs[0,:,:,0]
			binaryResult=gray2binary(probResult,0.35)
#			resultMerge=visualize([adjustImg,binaryResult],[1,2])

#			resultMerge=cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)

#			cv2.imwrite(self.config.test_result_path+self.analyze_name(path)+"_merge.jpg",resultMerge)
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.tif", (probResult*255).astype(np.uint8))
			#cv2.imwrite(self.config.test_result_path + self.analyze_name(path).replace('training','manual1') + ".tif", (binaryResult*255).astype(np.uint8))
			#np.save(self.config.test_result_path + self.analyze_name(path) + ".npy", binaryResult)


