"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""
import glob,cv2,numpy as np
import matplotlib.pyplot as plt
from perception.bases.data_loader_base import DataLoaderBase
from configs.utils.utils import write_hdf5,load_hdf5
import cv2

def CLAHE(img):
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    img = img.astype(np.uint8)
    return clahe.apply(img)

class MyDataLoader(DataLoaderBase):
	def __init__(self, config=None,dataset='CHASEDB1'):
		super(MyDataLoader, self).__init__(config)
		# 路径(data_path)、图片类型(img_type)
		self.train_img_path=config.train_img_path
		self.train_groundtruth_path = config.train_groundtruth_path
		self.prob_path = config.prob_path
		self.prob_val_path = config.prob_val_path
		self.train_type=config.train_datatype
		self.val_img_path=config.val_img_path
		self.val_groundtruth_path=config.val_groundtruth_path
		self.val_type = config.val_datatype

		# 实验名称(exp_name)
		self.exp_name=config.exp_name
		self.hdf5_path=config.hdf5_path
		self.height=config.height
		self.width=config.width
		self.num_seg_class=config.seg_num
		self.dataset = dataset
		self.G = self.config.G
		self.B = self.config.B
		self.R = self.config.R
		self.color_space = self.config.color_space
		self.multi_proportion = self.config.multi_proportion

	def _accesee_dataset_CHASEDB1(self,origin_path):
		"""

		:param origin_path:  原始图片路径(path for original image)
		:param groundtruth_path:  GT图片路径(path for groundtruth image)
		:return:  张量类型（Tensor） imgs， groundTruth
		"""
		orgList = glob.glob(origin_path+"*."+'jpg') #文件名列表 original image filename list
		gtList = glob.glob(origin_path+"*."+'png') #groundtruth 文件名列表 groundtruth image filename list
        
		print(origin_path)
		
		for num in range(len(orgList)):

           
			base_name = orgList[num].split('\\')[-1]

			gtList[num]=origin_path+base_name[:-4]+'_1stHO.png'
		assert (len(orgList) == len(gtList)) # 原始图片和GT图片数量应当一致 To make sure they have same length
		#print(gtList)
		imgs = np.empty((len(orgList), self.height, self.width, 1))
		groundTruth = np.empty((len(gtList), self.num_seg_class, self.height, self.width))

		for index in range(len(orgList)):
			orgPath=orgList[index]
			orgImg=plt.imread(orgPath)
#			orgImg = cv2.resize(orgImg,(565,584))
#			print(orgImg.shape)
#			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1]) 
			thresh = 1
			print('G_ratio:{} R_ratio:{}'.format(thresh,(1-thresh)))
			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1]*thresh+orgImg[:,:,0]*(1-thresh))   #血管在RGB图片的G通道非常明显，在B通道最不明显
#			g = orgImg[:,:,1]
#			r = orgImg[:,:,0]
#			clahe_g = CLAHE(g)
#			clahe_r = CLAHE(r)
#			imgs[index,:,:,0]=np.asarray(clahe_g*0.75 + clahe_r*0.25)
#			imgs[index,:,:,0]=np.asarray((CLAHE(orgImg[:,:,1]*0.75+orgImg[:,:,0]*0.25)))
			for no_seg in range(self.num_seg_class):
				gtPath=gtList[index]
				gtImg=cv2.imread(gtPath)
#				gtImg = cv2.resize(gtImg,(565,584))
#				print(gtImg.shape)
				if gtImg.shape[-1] == 3:
#                    pr
				    gtImg =cv2.cvtColor(gtImg,cv2.COLOR_RGB2GRAY)
				
				groundTruth[index,no_seg]=np.asarray(gtImg)
		print("[INFO] Reading...")
		assert (np.max(groundTruth) == 255)
		assert (np.min(groundTruth) == 0)
		return imgs,groundTruth        

	def _access_dataset(self,origin_path,groundtruth_path,datatype,mode):
		"""

		:param origin_path:  原始图片路径(path for original image)
		:param groundtruth_path:  GT图片路径(path for groundtruth image)
		:param datatype:  图片格式(dataType for origin and gt)
		:return:  张量类型（Tensor） imgs， groundTruth
		"""
		orgList = glob.glob(origin_path+"*."+datatype) #文件名列表 original image filename list
		gtList = glob.glob(groundtruth_path+"*."+datatype) #groundtruth 文件名列表 groundtruth image filename list
        
		print(origin_path)
		print(groundtruth_path)
		#有部分开发者反应，orglist与gtlist的文件名顺序在上一步骤后并不一一对应，所以添加下面的代码保证文件名的对应
		# Some Researchers find that filenames are not one-to-one match between orglist & gtlist,so I add the following part
		# 应根据训练数据的实际情况修改代码 please change the code according to your dataset filenames
		for num in range(len(orgList)):
#			loc=orgList[num].rfind('\\')  #此处可能出错，可以交换试验一下句是否可行   if this palce goes wrong,please switch to next line to have a try
			#loc=orgList[num].rfind('/')
#            print()
#			print(loc)  
#			print(orgList[num])
#			gtList[num]=groundtruth_path+orgList[num][loc+1:loc+4]+'manual1.tif'
			if mode == 'train':            
			    base_name = orgList[num].split('\\')[-1].replace('_training','_manual1')
			elif mode == 'val':
			    base_name = orgList[num].split('\\')[-1].replace('_test','_manual1')
#			print(base_name)
			gtList[num]=groundtruth_path+base_name
		assert (len(orgList) == len(gtList)) # 原始图片和GT图片数量应当一致 To make sure they have same length
		#print(gtList)
		if self.multi_proportion == True:
		    imgs = np.empty((len(orgList), self.height, self.width, 1))   
		else:            
		    imgs = np.empty((len(orgList), self.height, self.width, 3))
		groundTruth = np.empty((len(gtList), self.num_seg_class, self.height, self.width))
#		print('z')
		for index in range(len(orgList)):
#			print(1)
			orgPath=orgList[index]
			orgImg=plt.imread(orgPath)
#			orgImg =cv2.resize(orgImg,(999,960)) 
			print(imgs.shape)
			if self.color_space == 'RGB':
			    print('Color Space: RGB')                
			    if self.multi_proportion : 
			         print('ratio:{} R_ratio:{} B_ratio:{}'.format(self.G,self.R,self.B))            
			         imgs[index,:,:,0]=np.asarray(orgImg[:,:,1]*self.G+orgImg[:,:,0]*self.R+orgImg[:,:,2]*self.B)   #血管在RGB图片的G通道非常明显，在B通道最不明显
			    else:
    				 imgs[index,:,:]=np.asarray(orgImg)
			elif self.color_space == 'HSV':
			    print('Color Space: HSV')                 
			    orgImg = cv2.cvtColor(orgImg,cv2.COLOR_RGB2HSV)                
			    imgs[index,:,:]=np.asarray(orgImg)                

			elif self.color_space == 'LAB':
			    print('Color Space: LAB')                 
			    orgImg = cv2.cvtColor(orgImg,cv2.COLOR_RGB2LAB)                
			    imgs[index,:,:]=np.asarray(orgImg)                
#			thresh = 0.8
#			Blue_thresh = 0.1
#			print('G_ratio:{} R_ratio:{} B_ratio:{}'.format(thresh,(1-thresh-Blue_thresh),Blue_thresh))            
#			print('G_ratio:{} R_ratio:{} B_ratio:{}'.format(self.G,self.R,self.B))            
#			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1]*self.G+orgImg[:,:,0]*self.R+orgImg[:,:,2]*self.B)   #血管在RGB图片的G通道非常明显，在B通道最不明显
#			g = orgImg[:,:,1]
#			r = orgImg[:,:,0]
#			clahe_g = CLAHE(g)
#			clahe_r = CLAHE(r)
#			imgs[index,:,:,0]=np.asarray(clahe_g*0.75 + clahe_r*0.25)
#			imgs[index,:,:,0]=np.asarray((CLAHE(orgImg[:,:,1]*0.75+orgImg[:,:,0]*0.25)))
			for no_seg in range(self.num_seg_class):
#                print(2)
				gtPath=gtList[index]
				gtImg=plt.imread(gtPath,0)
#				gtImg = (gtImg > 128)*1*255
#				gtImg=cv2.resize(gtImg,(999,960))
				if gtImg.shape[-1] == 3:
#                    pr
				    gtImg =cv2.cvtColor(gtImg,cv2.COLOR_RGB2GRAY)
				    #gtImg = (gtImg > 128)*1*255
				    #print(gtImg)				
				groundTruth[index,no_seg]=np.asarray(gtImg)
		print("[INFO] Reading...")
		assert (np.max(groundTruth) == 255)
		assert (np.min(groundTruth) == 0)
		return imgs,groundTruth

	def _access_multiclass_dataset(self,origin_path,groundtruth_path,datatype,mode):
		"""

		:param origin_path:  原始图片路径(path for original image)
		:param groundtruth_path:  GT图片路径(path for groundtruth image)
		:param datatype:  图片格式(dataType for origin and gt)
		:return:  张量类型（Tensor） imgs， groundTruth
		"""
		orgList = glob.glob(origin_path+"*."+datatype) #文件名列表 original image filename list
		gtList = glob.glob(groundtruth_path+"*."+datatype) #groundtruth 文件名列表 groundtruth image filename list
        
		print(origin_path)
		print(groundtruth_path)
		#有部分开发者反应，orglist与gtlist的文件名顺序在上一步骤后并不一一对应，所以添加下面的代码保证文件名的对应
		# Some Researchers find that filenames are not one-to-one match between orglist & gtlist,so I add the following part
		# 应根据训练数据的实际情况修改代码 please change the code according to your dataset filenames
		for num in range(len(orgList)):
#			loc=orgList[num].rfind('\\')  #此处可能出错，可以交换试验一下句是否可行   if this palce goes wrong,please switch to next line to have a try
			#loc=orgList[num].rfind('/')
#            print()
#			print(loc)  
#			print(orgList[num])
#			gtList[num]=groundtruth_path+orgList[num][loc+1:loc+4]+'manual1.tif'
			if mode == 'train':            
			    base_name = orgList[num].split('\\')[-1].replace('_training','_manual1')
			elif mode == 'val':
			    base_name = orgList[num].split('\\')[-1].replace('_test','_manual1')
#			print(base_name)
			gtList[num]=groundtruth_path+base_name
		assert (len(orgList) == len(gtList)) # 原始图片和GT图片数量应当一致 To make sure they have same length
		#print(gtList)
		imgs = np.empty((len(orgList), self.height, self.width, 1))
		groundTruth = np.empty((len(gtList), self.num_seg_class, self.height, self.width))
#		print('z')
		for index in range(len(orgList)):
#			print(1)
			orgPath=orgList[index]
			orgImg=plt.imread(orgPath)
#			orgImg =cv2.resize(orgImg,(999,960)) 
#			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1])
			thresh = 1
			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1]*thresh+orgImg[:,:,0]*(1-thresh))   #血管在RGB图片的G通道非常明显，在B通道最不明显
#			g = orgImg[:,:,1]
#			r = orgImg[:,:,0]
#			clahe_g = CLAHE(g)
#			clahe_r = CLAHE(r)
#			imgs[index,:,:,0]=np.asarray(clahe_g*0.75 + clahe_r*0.25)
#			imgs[index,:,:,0]=np.asarray((CLAHE(orgImg[:,:,1]*0.75+orgImg[:,:,0]*0.25)))
			for no_seg in range(self.num_seg_class):

				gtPath=gtList[index]
				gtImg=plt.imread(gtPath,0)

			
				groundTruth[index,no_seg]=np.asarray(gtImg)
		print("[INFO] Reading...")
#		assert (np.max(groundTruth) == 255)
#		assert (np.min(groundTruth) == 0)
		return imgs,groundTruth


	def _access_STARE_dataset(self,origin_path,groundtruth_path,datatype,mode):
		"""

		:param origin_path:  原始图片路径(path for original image)
		:param groundtruth_path:  GT图片路径(path for groundtruth image)
		:param datatype:  图片格式(dataType for origin and gt)
		:return:  张量类型（Tensor） imgs， groundTruth
		"""
		orgList = glob.glob(origin_path+"*."+'bmp') #文件名列表 original image filename list
		gtList = glob.glob(groundtruth_path+"*."+'bmp') #groundtruth 文件名列表 groundtruth image filename list
        
		print(origin_path)
		print(groundtruth_path)
		#有部分开发者反应，orglist与gtlist的文件名顺序在上一步骤后并不一一对应，所以添加下面的代码保证文件名的对应
		# Some Researchers find that filenames are not one-to-one match between orglist & gtlist,so I add the following part
		# 应根据训练数据的实际情况修改代码 please change the code according to your dataset filenames
		for num in range(len(orgList)):
#			loc=orgList[num].rfind('\\')  #此处可能出错，可以交换试验一下句是否可行   if this palce goes wrong,please switch to next line to have a try
			#loc=orgList[num].rfind('/')
#            print()
#			print(loc)  
			print(orgList[num])
			#base_name=groundtruth_path+orgList[num]
#            print(gt)
#			gtList[num]=groundtruth_path+orgList[num][loc+1:loc+4]+'manual1.tif'
#			if mode == 'train':            
#			    base_name = orgList[num].split('\\')[-1].replace('_training','_manual1')
#			elif mode == 'val':
#			    base_name = orgList[num].split('\\')[-1].replace('_test','_manual1')
#			print(base_name)
			gtList[num]=orgList[num].replace('img_bmp','gt_bmp')
		assert (len(orgList) == len(gtList)) # 原始图片和GT图片数量应当一致 To make sure they have same length
		#print(gtList)
		imgs = np.empty((len(orgList), self.height, self.width, 1))
		groundTruth = np.empty((len(gtList), self.num_seg_class, self.height, self.width))

		for index in range(len(orgList)):
			orgPath=orgList[index]
			orgImg=plt.imread(orgPath)
#			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1])
			thresh = 0.8
			print('G_ratio:{} R_ratio:{}'.format(thresh,(1-thresh)))
			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1]*thresh+orgImg[:,:,0]*(1-thresh))   #血管在RGB图片的G通道非常明显，在B通道最不明显
#			g = orgImg[:,:,1]
#			r = orgImg[:,:,0]
#			clahe_g = CLAHE(g)
#			clahe_r = CLAHE(r)
#			imgs[index,:,:,0]=np.asarray(clahe_g*0.75 + clahe_r*0.25)
#			imgs[index,:,:,0]=np.asarray((CLAHE(orgImg[:,:,1]*0.75+orgImg[:,:,0]*0.25)))
			for no_seg in range(self.num_seg_class):
				gtPath=gtList[index]
				gtImg=plt.imread(gtPath,0)
				if gtImg.shape[-1] == 3:
#                    pr
				    gtImg =cv2.cvtColor(gtImg,cv2.COLOR_RGB2GRAY)
				
				groundTruth[index,no_seg]=np.asarray(gtImg)
		print("[INFO] Reading...")
		assert (np.max(groundTruth) == 255)
		assert (np.min(groundTruth) == 0)
		return imgs,groundTruth

	def _access_HRF_dataset(self,origin_path,groundtruth_path,datatype,mode):
		"""

		:param origin_path:  原始图片路径(path for original image)
		:param groundtruth_path:  GT图片路径(path for groundtruth image)
		:param datatype:  图片格式(dataType for origin and gt)
		:return:  张量类型（Tensor） imgs， groundTruth
		"""
		orgList = glob.glob(origin_path+"*."+'jpg') #文件名列表 original image filename list
		gtList = glob.glob(groundtruth_path+"*."+'tif') #groundtruth 文件名列表 groundtruth image filename list
        
		print(origin_path)
		print(groundtruth_path)
		#有部分开发者反应，orglist与gtlist的文件名顺序在上一步骤后并不一一对应，所以添加下面的代码保证文件名的对应
		# Some Researchers find that filenames are not one-to-one match between orglist & gtlist,so I add the following part
		# 应根据训练数据的实际情况修改代码 please change the code according to your dataset filenames
		for num in range(len(orgList)):
#			loc=orgList[num].rfind('\\')  #此处可能出错，可以交换试验一下句是否可行   if this palce goes wrong,please switch to next line to have a try
			#loc=orgList[num].rfind('/')
#            print()
#			print(loc)  
			print(orgList[num])
			#base_name=groundtruth_path+orgList[num]
#            print(gt)
#			gtList[num]=groundtruth_path+orgList[num][loc+1:loc+4]+'manual1.tif'
#			if mode == 'train':            
#			    base_name = orgList[num].split('\\')[-1].replace('_training','_manual1')
#			elif mode == 'val':
#			    base_name = orgList[num].split('\\')[-1].replace('_test','_manual1')
#			print(base_name)
			gtList[num]= orgList[num].replace('images','gt').replace('JPG','tif').replace('jpg','tif')
		assert (len(orgList) == len(gtList)) # 原始图片和GT图片数量应当一致 To make sure they have same length
		#print(gtList)
		imgs = np.empty((len(orgList), self.height, self.width, 1))
		groundTruth = np.empty((len(gtList), self.num_seg_class, self.height, self.width))

		for index in range(len(orgList)):
			orgPath=orgList[index]
#			orgPath[-4:] = '.tif'
#			print(orgPath)
			orgImg=plt.imread(orgPath)
			orgImg = cv2.resize(orgImg,(1727,1168))
#			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1])
			thresh = 1
			print('G_ratio:{} R_ratio:{}'.format(thresh,(1-thresh)))
			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1]*thresh+orgImg[:,:,0]*(1-thresh))   #血管在RGB图片的G通道非常明显，在B通道最不明显
#			g = orgImg[:,:,1]
#			r = orgImg[:,:,0]
#			clahe_g = CLAHE(g)
#			clahe_r = CLAHE(r)
#			imgs[index,:,:,0]=np.asarray(clahe_g*0.75 + clahe_r*0.25)
#			imgs[index,:,:,0]=np.asarray((CLAHE(orgImg[:,:,1]*0.75+orgImg[:,:,0]*0.25)))
			for no_seg in range(self.num_seg_class):
				gtPath=gtList[index]
				gtImg=plt.imread(gtPath,0)
				gtImg = cv2.resize(gtImg,(1727,1168))
				if gtImg.shape[-1] == 3:
#                    pr
				    gtImg =cv2.cvtColor(gtImg,cv2.COLOR_RGB2GRAY)
				
				groundTruth[index,no_seg]=np.asarray(gtImg)
		print("[INFO] Reading...")
		assert (np.max(groundTruth) == 255)
		assert (np.min(groundTruth) == 0)
		return imgs,groundTruth

	def _access_prob_dataset(self,origin_path,groundtruth_path,prob_path,datatype,mode):
		"""

		:param origin_path:  原始图片路径(path for original image)
		:param groundtruth_path:  GT图片路径(path for groundtruth image)
		:param datatype:  图片格式(dataType for origin and gt)
		:return:  张量类型（Tensor） imgs， groundTruth
		"""
		orgList = glob.glob(origin_path+"*."+datatype) #文件名列表 original image filename list
		gtList = glob.glob(groundtruth_path+"*."+datatype) #groundtruth 文件名列表 groundtruth image filename list
		probList = glob.glob(prob_path+"*."+datatype) #groundtruth 文件名列表 groundtruth image filename list
        
		print(origin_path)
		print(groundtruth_path)
		#有部分开发者反应，orglist与gtlist的文件名顺序在上一步骤后并不一一对应，所以添加下面的代码保证文件名的对应
		# Some Researchers find that filenames are not one-to-one match between orglist & gtlist,so I add the following part
		# 应根据训练数据的实际情况修改代码 please change the code according to your dataset filenames
		for num in range(len(orgList)):
#			loc=orgList[num].rfind('\\')  #此处可能出错，可以交换试验一下句是否可行   if this palce goes wrong,please switch to next line to have a try
			#loc=orgList[num].rfind('/')
#            print()
#			print(loc)  
#			print(orgList[num])
#			gtList[num]=groundtruth_path+orgList[num][loc+1:loc+4]+'manual1.tif'
			if mode == 'train':            
			    base_name = orgList[num].split('\\')[-1].replace('_training','_manual1')
			elif mode == 'val':
			    base_name = orgList[num].split('\\')[-1].replace('_test','_manual1')
			elif mode == 'prob':
			    base_name = orgList[num].split('\\')[-1].replace('training','training_prob')
			elif mode == 'val_prob':
			    base_name = orgList[num].split('\\')[-1].replace('test','test_prob')

#			print(base_name)
			gtList[num]=groundtruth_path+base_name
			probList[num]=prob_path+base_name
		assert (len(orgList) == len(gtList)) # 原始图片和GT图片数量应当一致 To make sure they have same length
		#print(gtList)
		imgs = np.empty((len(orgList), self.height, self.width, 1))
		probmap = np.empty((len(probList), self.num_seg_class, self.height, self.width))

		for index in range(len(orgList)):
			orgPath=orgList[index]
			orgImg=plt.imread(orgPath)

			thresh = 1
			imgs[index,:,:,0]=np.asarray(orgImg[:,:,1]*thresh+orgImg[:,:,0]*(1-thresh))   #血管在RGB图片的G通道非常明显，在B通道最不明显
#
#			for no_seg in range(self.num_seg_class):
#				gtPath=gtList[index]
#				gtImg=plt.imread(gtPath,0)
##				gtImg=cv2.resize(gtImg,(999,960))
#				if gtImg.shape[-1] == 3:
##                    pr
#				    gtImg =cv2.cvtColor(gtImg,cv2.COLOR_RGB2GRAY)
#				
#				groundTruth[index,no_seg]=np.asarray(gtImg)
			
			for no_seg in range(self.num_seg_class):
				probPath=probList[index]
				probImg=plt.imread(probPath,0)
#				gtImg=cv2.resize(gtImg,(999,960))
				if probImg.shape[-1] == 3:
#                    pr
				    probImg =cv2.cvtColor(probImg,cv2.COLOR_RGB2GRAY)
				
				probmap[index,no_seg]=np.asarray(probImg)
		print("[INFO] Reading...")

		return probmap
    
    
	def prepare_dataset(self):
		if self.dataset == 'CHASEDB1':
		    print('Writing CHASEDB1....')
		    imgs_train, train_groundTruth=self._accesee_dataset_CHASEDB1(self.train_img_path)
		    imgs_val, val_groundTruth = self._accesee_dataset_CHASEDB1(self.val_img_path)
		elif self.dataset == 'DRIVE':
		    print('Writing DRIVE....')
		    imgs_train, train_groundTruth=self._access_dataset(self.train_img_path,self.train_groundtruth_path,self.train_type,'train')
		    imgs_val, val_groundTruth = self._access_dataset(self.val_img_path,self.val_groundtruth_path,self.val_type,'val')
		
		elif self.dataset == 'DRIVE_MULTICLASS':
		    print('Writing DRIVE_MULTICALSS....')
		    imgs_train, train_groundTruth=self._access_multiclass_dataset(self.train_img_path,self.train_groundtruth_path,self.train_type,'train')
		    imgs_val, val_groundTruth = self._access_multiclass_dataset(self.val_img_path,self.val_groundtruth_path,self.val_type,'val')

		elif self.dataset == 'STARE':
		    print('Writing STARE....')
		    imgs_train, train_groundTruth=self._access_STARE_dataset(self.train_img_path,self.train_groundtruth_path,self.train_type,'train')
		    imgs_val, val_groundTruth = self._access_STARE_dataset(self.val_img_path,self.val_groundtruth_path,self.val_type,'val')
		elif self.dataset=='CROSS':
		    imgs_train, train_groundTruth=self._access_dataset(self.train_img_path,self.train_groundtruth_path,self.train_type,'train')
		    imgs_val, val_groundTruth = self._accesee_dataset_CHASEDB1(self.val_img_path)
		elif self.dataset == 'HRF':
		    imgs_train, train_groundTruth=self._access_HRF_dataset(self.train_img_path,self.train_groundtruth_path,self.train_type,'train')
		    imgs_val, val_groundTruth = self._access_HRF_dataset(self.val_img_path,self.val_groundtruth_path,self.val_type,'val')

		elif self.dataset == 'prob':
		    imgs_train, train_groundTruth=self._access_dataset(self.train_img_path,self.train_groundtruth_path,self.train_type,'train')
		    imgs_val, val_groundTruth = self._access_dataset(self.val_img_path,self.val_groundtruth_path,self.val_type,'val')
		    probs_train = self._access_prob_dataset(self.train_img_path,self.train_groundtruth_path,self.prob_path,self.val_type,'prob')
		    probs_val = self._access_prob_dataset(self.val_img_path,self.val_groundtruth_path,self.prob_val_path,self.val_type,'val_prob')
          
#		    imgs_train, train_groundTruth=self._accesee_dataset_CHASEDB1(self.train_img_path)
#		    imgs_val, val_groundTruth = self._access_dataset(self.val_img_path,self.val_groundtruth_path,self.val_type,'val')
            
		write_hdf5(imgs_train,self.hdf5_path+"/_"+self.dataset+"_train_img.hdf5")
		write_hdf5(train_groundTruth, self.hdf5_path+"/_"+self.dataset+"_train_groundtruth.hdf5")
		print("[INFO] Saving Training Data")
		# 测试图片汇成HDF5合集 preapare val_img/groundtruth.hdf5
#		    imgs_val, groundTruth = self._accesee_dataset_CHASEDB1(self.val_img_path)
		write_hdf5(imgs_val, self.hdf5_path + "/_"+self.dataset+"_val_img.hdf5")
		write_hdf5(val_groundTruth, self.hdf5_path + "/_"+self.dataset+"_val_groundtruth.hdf5")
		print("[INFO] Saving Validation Data")
        
		if self.dataset =='prob':
		    write_hdf5(probs_val,self.hdf5_path+"/_"+self.dataset+"_prob_val_img.hdf5")
            
		    write_hdf5(probs_train,self.hdf5_path+"/_"+self.dataset+"_prob_img.hdf5")
#		write_hdf5(train_groundTruth, self.hdf5_path+"/_"+self.dataset+"_train_groundtruth.hdf5")
		    print("[INFO] Saving prob Data")        
		# 训练图片汇成HDF5合集 preapare train_img/groundtruth.hdf5
#        
#		else:
#            
#		    imgs_train, groundTruth=self._access_dataset(self.train_img_path,self.train_groundtruth_path,self.train_type,'train')
#		    write_hdf5(imgs_train,self.hdf5_path+"/train_img.hdf5")
#		    write_hdf5(groundTruth, self.hdf5_path+"/train_groundtruth.hdf5")
#		    print("[INFO] Saving Training Data")
#    		# 测试图片汇成HDF5合集 preapare val_img/groundtruth.hdf5
#		    imgs_val, groundTruth = self._access_dataset(self.val_img_path, self.val_groundtruth_path, self.val_type,'val')
#		    write_hdf5(imgs_val, self.hdf5_path + "/val_img.hdf5")
#		    write_hdf5(groundTruth, self.hdf5_path + "/val_groundtruth.hdf5")
#		    print("[INFO] Saving Validation Data")

	def get_train_data(self):
       #if self_dataset == 'CHASEDB1':
            
		imgs_train=load_hdf5(self.hdf5_path+"/_"+self.dataset+"_train_img.hdf5")
		groundTruth=load_hdf5(self.hdf5_path+"/_"+self.dataset+"_train_groundtruth.hdf5")
		return imgs_train,groundTruth

	def get_val_data(self):
		imgs_val=load_hdf5(self.hdf5_path + "/_"+self.dataset+"_val_img.hdf5")
		groundTruth=load_hdf5(self.hdf5_path + "/_"+self.dataset+"_val_groundtruth.hdf5")
		return imgs_val,groundTruth
    
	def get_prob_data(self):
       #if self_dataset == 'CHASEDB1':
            
		probs_train=load_hdf5(self.hdf5_path+"/_"+self.dataset+"_prob_img.hdf5")
		probs_vals=load_hdf5(self.hdf5_path+"/_"+self.dataset+"_prob_val_img.hdf5")
		#groundTruth=load_hdf5(self.hdf5_path+"/_"+self.dataset+"_train_groundtruth.hdf5")
		return probs_train,probs_vals
   
#    
#for i in range(1):
#    print(i)