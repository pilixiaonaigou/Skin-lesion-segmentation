import random,numpy as np,cv2
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback,ReduceLROnPlateau
from keras import callbacks 
from perception.bases.trainer_base import TrainerBase
from configs.utils.utils import genMasks,visualize,genMasks2
from configs.utils.img_utils import img_process,img_process1
import keras
global gen

#num_list_1 = []
#num_list_2 = []
class SWA(keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')

class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1,iter_=0):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr
        self.iter_ = iter_

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            callbacks.ModelCheckpoint("./keras_{}.model".format(self.iter_),monitor='val_loss', 
                                   mode = 'min', save_best_only=True, verbose=1),
#            SWA('./keras_384_2.model',1),
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)


class SegmentionTrainer(TrainerBase):
	def __init__(self,model,data,config,iter_):
		super(SegmentionTrainer, self).__init__(model, data, config , iter_)
		self.model=model
		self.data=data
		self.config=config
		self.callbacks=[]
		self.init_callbacks()
		self.iter_ = iter_

	def init_callbacks(self):
		self.callbacks = SnapshotCallbackBuilder(nb_epochs=2,nb_snapshots=1,init_lr=1e-3,iter_=self.iter_).get_callbacks()
#		self.callbacks.append(
#			ModelCheckpoint(
#				filepath=self.config.hdf5_path+self.config.exp_name+ '_best_weights.h5',
#		        verbose=1,
#		        monitor='val_loss',
#		        mode='auto',
#		        save_best_only=True
#			)
#		)
#		self.callbacks.append(
#                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto' 
#                        )
#                )
#
#		self.callbacks.append(
#			TensorBoard(
#				log_dir=self.config.checkpoint,
#				write_images=True,
#				write_graph=True,
#			)
#		)

	def train(self):
		gen=MulticlassDataGenerator(self.data,self.config,self.iter_)
		if self.iter_ =='DRIVE_Binary':
        
		    gen=DataGeneratorBinary(self.data,self.config,self.iter_)
		elif self.iter_ =='DROIVE_MULTICLASS':
		    gen=MulticlassDataGenerator(self.data,self.config,self.iter_)
		#gen.visual_patch()
		hist = self.model.fit_generator(gen.train_gen(),
		    epochs=self.config.epochs,
            steps_per_epoch=100,
#		    steps_per_epoch=self.config.subsample * self.config.total_train / self.config.batch_size,
		    verbose=1,
		    callbacks=self.callbacks,
		    validation_data=gen.val_gen(),
			validation_steps=int(self.config.subsample * self.config.total_val / self.config.batch_size)
		)
		self.model.save_weights(self.config.hdf5_path+self.config.exp_name+'_last_weights.h5', overwrite=True)

class MulticlassDataGenerator():
	"""
	load image (Generator)
	"""
	def __init__(self,data,config,iter_):
		if iter_ == 'prob' :
			self.train_img=img_process1(data[0],data[4])
			self.val_img=img_process1(data[2],data[5])
		
		elif iter_ == 'DRIVE_MULTICLASS' :
			print('DRIVE_MULTICLASS')
			self.train_img=img_process(data[0])
			self.val_img=img_process(data[2])
			self.train_gt=data[1]
#    		self.val_img=img_process(data[2])
#		self.val_img=data[2]
			self.val_gt=data[3]               
		else:
            
			self.train_img=img_process(data[0])
			self.val_img=img_process(data[2])        
#		self.train_img=data[0]
    		#self.train_gt=data[1]/255.
    #    		self.val_img=img_process(data[2])
    #		self.val_img=data[2]
    		#self.val_gt=data[3]/255.
		self.config=config
		self.multi_proportion = config.multi_proportion
	def _CenterSampler(self,attnlist,class_weight,Nimgs):
		"""
		围绕目标区域采样
		:param attnlist:  目标区域坐标
		:return: 采样的坐标
		"""
		class_weight = class_weight / np.sum(class_weight)
		p = random.uniform(0, 1)
		psum = 0
		for i in range(class_weight.shape[0]):
			psum = psum + class_weight[i]
			if p < psum:
				label = i
				break
		if label == class_weight.shape[0] - 1:
			i_center = random.randint(0, Nimgs - 1)
			x_center = random.randint(0 + int(self.config.patch_width / 2), self.config.width - int(self.config.patch_width / 2))
			# print "x_center " +str(x_center)
			y_center = random.randint(0 + int(self.config.patch_height / 2), self.config.height - int(self.config.patch_height / 2))
		else:
			t = attnlist[label]
			cid = random.randint(0, t[0].shape[0] - 1)
			i_center = t[0][cid]
			y_center = t[1][cid] + random.randint(0 - int(self.config.patch_width / 2), 0 + int(self.config.patch_width / 2))
			x_center = t[2][cid] + random.randint(0 - int(self.config.patch_width / 2), 0 + int(self.config.patch_width / 2))

		if y_center < self.config.patch_width / 2:
			y_center = self.config.patch_width / 2
		elif y_center > self.config.height - self.config.patch_width / 2:
			y_center = self.config.height - self.config.patch_width / 2

		if x_center < self.config.patch_width / 2:
			x_center = self.config.patch_width / 2
		elif x_center > self.config.width - self.config.patch_width / 2:
			x_center = self.config.width - self.config.patch_width / 2

		return i_center, x_center, y_center

	def _genDef(self,train_imgs,train_masks,attnlist,class_weight):
		"""
		图片取块生成器模板
		:param train_imgs: 原始图
		:param train_masks:  原始图groundtruth
		:param attnlist:  目标区域list
		:return:  取出的训练样本
		"""
		while 1:
			Nimgs=train_imgs.shape[0]
			for t in range(int(self.config.subsample * self.config.total_train / self.config.batch_size)):
				if self.multi_proportion:                    
					X = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,1])
					Y = np.zeros([self.config.batch_size, self.config.patch_height * self.config.patch_width, self.config.seg_num + 1])
				else:
					X = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,3])
					Y = np.zeros([self.config.batch_size, self.config.patch_height * self.config.patch_width, self.config.seg_num + 1])

				for j in range(self.config.batch_size):
					[i_center, x_center, y_center] = self._CenterSampler(attnlist,class_weight,Nimgs)
					patch = train_imgs[i_center, int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2),:]
					patch_mask = train_masks[i_center, :, int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2)]
					X[j, :, :, :] = patch
					Y[j, :, :] = genMasks2(np.reshape(patch_mask, [1, self.config.seg_num, self.config.patch_height, self.config.patch_width]),self.config.seg_num)
				yield (X, Y)

	def train_gen(self):
		"""
		训练样本生成器
		"""
		class_weight=[1.0,0.0]
		attnlist=[np.where(self.train_gt[:,0,:,:]==np.max(self.train_gt[:,0,:,:]))]
		return self._genDef(self.train_img,self.train_gt,attnlist,class_weight)

	def val_gen(self):
		"""
		验证样本生成器
		"""
		class_weight = [1.0,0.0]
		attnlist = [np.where(self.val_gt[:, 0, :, :] == np.max(self.val_gt[:, 0, :, :]))]
		return self._genDef(self.val_img, self.val_gt, attnlist,class_weight)

	def visual_patch(self):
		gen=self.train_gen()
		(X,Y)=next(gen)
		image=[]
		mask=[]
		print("[INFO] Visualize Image Sample...")
		for index in range(self.config.batch_size):
			image.append(X[index])
			mask.append(np.reshape(Y,[self.config.batch_size,self.config.patch_height,self.config.patch_width,self.config.seg_num+1])[index,:,:,0])
		if self.config.batch_size%4==0:
			row=self.config.batch_size/4
			col=4
		else:
			if self.config.batch_size % 5!=0:
				row = self.config.batch_size // 5+1
			else:
				row = self.config.batch_size // 5
			col = 5
		imagePatch=visualize(image,[row,col])
		maskPatch=visualize(mask,[row,col])
		cv2.imwrite(self.config.checkpoint+"image_patch.jpg",imagePatch)
		cv2.imwrite(self.config.checkpoint + "groundtruth_patch.jpg", maskPatch)




class DataGenerator():
	"""
	load image (Generator)
	"""
	def __init__(self,data,config,iter_):
		if iter_ == 'prob' :
			self.train_img=img_process1(data[0],data[4])
			self.val_img=img_process1(data[2],data[5])
		else:
            
			self.train_img=img_process(data[0])
			self.val_img=img_process(data[2])        
#		self.train_img=data[0]
		self.train_gt=data[1]/255.
#    		self.val_img=img_process(data[2])
#		self.val_img=data[2]
		self.val_gt=data[3]/255.
		self.config=config
		self.multi_proportion = config.multi_proportion
	def _CenterSampler(self,attnlist,class_weight,Nimgs):
		"""
		围绕目标区域采样
		:param attnlist:  目标区域坐标
		:return: 采样的坐标
		"""
		class_weight = class_weight / np.sum(class_weight)
		p = random.uniform(0, 1)
		psum = 0
		for i in range(class_weight.shape[0]):
			psum = psum + class_weight[i]
			if p < psum:
				label = i
				break
		if label == class_weight.shape[0] - 1:
			i_center = random.randint(0, Nimgs - 1)
			x_center = random.randint(0 + int(self.config.patch_width / 2), self.config.width - int(self.config.patch_width / 2))
			# print "x_center " +str(x_center)
			y_center = random.randint(0 + int(self.config.patch_height / 2), self.config.height - int(self.config.patch_height / 2))
		else:
			t = attnlist[label]
			cid = random.randint(0, t[0].shape[0] - 1)
			i_center = t[0][cid]
			y_center = t[1][cid] + random.randint(0 - int(self.config.patch_width / 2), 0 + int(self.config.patch_width / 2))
			x_center = t[2][cid] + random.randint(0 - int(self.config.patch_width / 2), 0 + int(self.config.patch_width / 2))

		if y_center < self.config.patch_width / 2:
			y_center = self.config.patch_width / 2
		elif y_center > self.config.height - self.config.patch_width / 2:
			y_center = self.config.height - self.config.patch_width / 2

		if x_center < self.config.patch_width / 2:
			x_center = self.config.patch_width / 2
		elif x_center > self.config.width - self.config.patch_width / 2:
			x_center = self.config.width - self.config.patch_width / 2

		return i_center, x_center, y_center

	def _genDef(self,train_imgs,train_masks,attnlist,class_weight):
		"""
		图片取块生成器模板
		:param train_imgs: 原始图
		:param train_masks:  原始图groundtruth
		:param attnlist:  目标区域list
		:return:  取出的训练样本
		"""
		while 1:
			Nimgs=train_imgs.shape[0]
			for t in range(int(self.config.subsample * self.config.total_train / self.config.batch_size)):
				if self.multi_proportion:                    
					X = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,1])
					Y = np.zeros([self.config.batch_size, self.config.patch_height * self.config.patch_width, self.config.seg_num + 1])
				else:
					X = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,3])
					Y = np.zeros([self.config.batch_size, self.config.patch_height * self.config.patch_width, self.config.seg_num + 1])
				for j in range(self.config.batch_size):
					[i_center, x_center, y_center] = self._CenterSampler(attnlist,class_weight,Nimgs)
					patch = train_imgs[i_center, int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2),:]
					patch_mask = train_masks[i_center, :, int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2)]
					X[j, :, :, :] = patch
					Y[j, :, :] = genMasks(np.reshape(patch_mask, [1, self.config.seg_num, self.config.patch_height, self.config.patch_width]),self.config.seg_num)
				yield (X, Y)

	def train_gen(self):
		"""
		训练样本生成器
		"""
		class_weight=[1.0,0.0]
		attnlist=[np.where(self.train_gt[:,0,:,:]==np.max(self.train_gt[:,0,:,:]))]
		return self._genDef(self.train_img,self.train_gt,attnlist,class_weight)

	def val_gen(self):
		"""
		验证样本生成器
		"""
		class_weight = [1.0,0.0]
		attnlist = [np.where(self.val_gt[:, 0, :, :] == np.max(self.val_gt[:, 0, :, :]))]
		return self._genDef(self.val_img, self.val_gt, attnlist,class_weight)

	def visual_patch(self):
		gen=self.train_gen()
		(X,Y)=next(gen)
		image=[]
		mask=[]
		print("[INFO] Visualize Image Sample...")
		for index in range(self.config.batch_size):
			image.append(X[index])
			mask.append(np.reshape(Y,[self.config.batch_size,self.config.patch_height,self.config.patch_width,self.config.seg_num+1])[index,:,:,0])
		if self.config.batch_size%4==0:
			row=self.config.batch_size/4
			col=4
		else:
			if self.config.batch_size % 5!=0:
				row = self.config.batch_size // 5+1
			else:
				row = self.config.batch_size // 5
			col = 5
		imagePatch=visualize(image,[row,col])
		maskPatch=visualize(mask,[row,col])
		cv2.imwrite(self.config.checkpoint+"image_patch.jpg",imagePatch)
		cv2.imwrite(self.config.checkpoint + "groundtruth_patch.jpg", maskPatch)



class DataGeneratorBinary():
	"""
	load image (Generator)
	"""
	def __init__(self,data,config,iter_):
		if iter_ == 'prob' :
			self.train_img=img_process1(data[0],data[4])
			self.val_img=img_process1(data[2],data[5])
			self.train_gt=data[1]
#    		self.val_img=img_process(data[2])
#		self.val_img=data[2]
			self.val_gt=data[3]       
            
		elif iter_ == 'DRIVE_MULTICLASS' :
			self.train_img=img_process1(data[0],data[4])
			self.val_img=img_process1(data[2],data[5])
			self.train_gt=data[1]
#    		self.val_img=img_process(data[2])
#		self.val_img=data[2]
			self.val_gt=data[3]              
		else:
            
			self.train_img=img_process(data[0])
			self.val_img=img_process(data[2])        
#		self.train_img=data[0]
		self.train_gt=data[1]/255.
#    		self.val_img=img_process(data[2])
#		self.val_img=data[2]
		self.val_gt=data[3]/255.
		self.config=config
		self.multi_proportion = config.multi_proportion
	def _CenterSampler(self,attnlist,class_weight,Nimgs,num_list):
		"""
		围绕目标区域采样
		:param attnlist:  目标区域坐标
		:return: 采样的坐标
		"""
		class_weight = class_weight / np.sum(class_weight)
		p = random.uniform(0, 1)
		psum = 0
		for i in range(class_weight.shape[0]):
			psum = psum + class_weight[i]
			if p < psum:
				label = i
				break
		if label == class_weight.shape[0] - 1:
#			print('Nimg:{}'.format(Nimgs))
			i_center = random.randint(0, Nimgs - 1)
			x_center = random.randint(0 + int(self.config.patch_width // 2), self.config.width - int(self.config.patch_width // 2))
			# print "x_center " +str(x_center)
			y_center = random.randint(0 + int(self.config.patch_height // 2), self.config.height - int(self.config.patch_height // 2))
		else:
			t = attnlist[label]
#			while 1:
                
#			    cid = random.randint(0, t[0].shape[0] - 1)
#                
#			    if cid not in num_list:
#			        num_list.append(cid)
#			        break
                    
#			cid = num_id
#			print('label:{}'.format(label))
#			print('cid:{}'.format(t[0].shape[0]))
			cid = random.randint(0, t[0].shape[0] - 1)    
			i_center = t[0][cid]
			y_center = t[1][cid] + random.randint(0 - int(self.config.patch_width // 2), 0 + int(self.config.patch_width // 2))
			x_center = t[2][cid] + random.randint(0 - int(self.config.patch_width // 2), 0 + int(self.config.patch_width // 2))
#			print('初始点:{} {}'.format(x_center,y_center))
		if y_center < self.config.patch_width // 2:
			y_center = self.config.patch_width // 2
		elif y_center > self.config.height - self.config.patch_width // 2:
			y_center = self.config.height - self.config.patch_width // 2

		if x_center < self.config.patch_width // 2:
			x_center = self.config.patch_width // 2
		elif x_center > self.config.width - self.config.patch_width // 2:
			x_center = self.config.width - self.config.patch_width // 2
		
#		print('修正后初始点:{} {}'.format(x_center,y_center))
		return i_center, x_center, y_center

	def _genDef(self,train_imgs,train_masks,attnlist,class_weight,num_list):
		"""
		图片取块生成器模板
		:param train_imgs: 原始图
		:param train_masks:  原始图groundtruth
		:param attnlist:  目标区域list
		:return:  取出的训练样本
		"""
#		num_id = 0
		while 1:
#			num_id = 0
			Nimgs=train_imgs.shape[0]
#			print(train_imgs.shape)
#            num_ = at
			for t in range(int(self.config.subsample * self.config.total_train / self.config.batch_size)):
				if self.multi_proportion:                    
					X = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,1])
					Y = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,1])
				else:
					X = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,3])
					Y = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,1])

#					Y = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,1])

				for j in range(self.config.batch_size):
					[i_center, x_center, y_center] = self._CenterSampler(attnlist,class_weight,Nimgs,num_list)
#					num_id += 1
#					if num_id >= 925896:
#                        num_id = 0
					patch = train_imgs[i_center, int(y_center - self.config.patch_height // 2):int(y_center + self.config.patch_height // 2),int(x_center - self.config.patch_width // 2):int(x_center + self.config.patch_width // 2),:]
					patch_mask = train_masks[i_center, :, int(y_center - self.config.patch_height // 2):int(y_center + self.config.patch_height // 2),int(x_center - self.config.patch_width // 2):int(x_center + self.config.patch_width // 2)]
#					patch = np.expand_dims(cv2.resize(patch,(self.config.patch_height,self.config.patch_width)),-1)
#					patch_mask = cv2.resize(patch_mask,(self.config.patch_height,self.config.patch_width))
					X[j, :, :, :] = patch
#					Y[j, :, :] = genMasks(np.reshape(patch_mask, [1, self.config.seg_num, self.config.patch_height, self.config.patch_width]),self.config.seg_num)
					Y[j, :, :] = np.transpose(patch_mask,(1,2,0))
				yield (X, Y)

	def train_gen(self):
		"""
		训练样本生成器
		"""
		class_weight=[1.0,0.0]
		train_num_list = []
		attnlist=[np.where(self.train_gt[:,0,:,:]==np.max(self.train_gt[:,0,:,:]))]
		return self._genDef(self.train_img,self.train_gt,attnlist,class_weight,train_num_list)

	def val_gen(self):
		"""
		验证样本生成器
		"""
		class_weight = [1.0,0.0]
		attnlist = [np.where(self.val_gt[:, 0, :, :] == np.max(self.val_gt[:, 0, :, :]))]
		val_num_list = []
		return self._genDef(self.val_img, self.val_gt, attnlist,class_weight,val_num_list)

	def visual_patch(self):
		gen=self.train_gen()
		(X,Y)=next(gen)
		image=[]
		mask=[]
		print("[INFO] Visualize Image Sample...")
		for index in range(self.config.batch_size):
			image.append(X[index])
			mask.append(np.reshape(Y,[self.config.batch_size,self.config.patch_height,self.config.patch_width,self.config.seg_num+1])[index,:,:,0])
		if self.config.batch_size%4==0:
			row=self.config.batch_size/4
			col=4
		else:
			if self.config.batch_size % 5!=0:
				row = self.config.batch_size // 5+1
			else:
				row = self.config.batch_size // 5
			col = 5
		imagePatch=visualize(image,[row,col])
		maskPatch=visualize(mask,[row,col])
		cv2.imwrite(self.config.checkpoint+"image_patch.jpg",imagePatch)
		cv2.imwrite(self.config.checkpoint + "groundtruth_patch.jpg", maskPatch)