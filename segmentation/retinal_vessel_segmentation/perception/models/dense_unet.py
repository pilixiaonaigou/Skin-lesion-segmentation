"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""
from keras import backend as K
import keras,os
from keras.models import Model
from keras.layers.merge import add,multiply,average,dot
from keras.layers import BatchNormalization, Add,Lambda,Input, Conv2D,Conv2DTranspose,MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,normalization,concatenate,Activation,Reshape,multiply,GlobalAveragePooling2D,Dense
from keras import backend as K
from keras.layers.core import Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.utils import plot_model
#from keras.optmizers import SGD
from perception.bases.model_base import ModelBase
from keras.optimizers import SGD
from keras.utils.generic_utils import get_custom_objects
from keras.legacy import interfaces
from keras import optimizers
import tensorflow as tf
from keras.losses import categorical_crossentropy,binary_crossentropy
from deform_conv.layers import  ConvOffset2D
# https://github.com/titu1994/keras-normalized-optimizers
# Computes the L-2 norm of the gradient.


alpha=0.25, 
gamma=2.0

def binary_focal_loss(y_true, y_pred):
    # # # filter out "ignore" anchors
    # anchor_state = K.max(y_true, axis=2)  # -1 for ignore, 0 for background, 1+ for objects
    # indices = tf.where(K.not_equal(anchor_state, -1))
    # y_true = tf.gather_nd(y_true, indices)
    # y_pred = tf.gather_nd(y_pred, indices)

    # compute the focal loss
    # CE(p_t) = -log(p_t)
    # FL(p_t) = -(1 - p_t) ** gamma * log(p_t)
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma

    loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
    return loss

    # # compute the normalizer: the number of positive anchors
    # normalizer = tf.where(K.equal(anchor_state, 1))
    # normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
    # normalizer = K.maximum(1., normalizer)
    # return K.sum(cls_loss) / normalizer

def categorical_focal_loss(y_true, y_pred):
    alpha_factor = K.ones_like(y_true) * alpha
    
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1. - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1. - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma

    loss = focal_weight * categorical_crossentropy(y_true, y_pred)
    normalizer = K.sum(K.abs(y_true), axis=[1, 2])
    loss = K.sum(loss, axis=[1, 2])/K.maximum(1., normalizer)
    return K.mean(loss)



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float32')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bin_loss_with_L1(y_true,y_pred):
    
#    target = K.flatten(y_ture)
#    return binary_crossentropy(y_true,y_pred)+ K.mean(K.abs(y_pred-y_true),axis=-1)
    return binary_crossentropy(y_true,y_pred)
#def focal_loss(gamma=2., alpha=.25):


#def Mixed_Loss(gamma=2,alpha =0.25,factor=8):
#	def focal_loss_fixed(y_true, y_pred):
#		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#		return (10-factor)*(-K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))) + factor*categorical_crossentropy(y_true,y_pred)
#	return  focal_loss_fixed
#    
    
    
def l2_norm(grad):
    norm = K.sqrt(K.sum(K.square(grad))) + K.epsilon()
    return norm


def BatchActivate(x):
    x = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(x)
#    x1 = Activation('sigmoid')(x)
    x = Activation('relu')(x)
#    return add([multiply([x1,x]),x2])
#    x =PReLU()(x)
    return x

def ConvBlock(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters,size,strides=strides,padding=padding)(x)
    x = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(x)
#    x = BatchNormalization()(x)
#    x = PReLU()(x)
    x = Activation('relu')(x)
    return x

    

def GatedBlock(x):
    x = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(x)
    x1 = Activation('sigmoid')(x)
    x2 = PReLU()(x)
    return add([multiply([x1,x]),x2])


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True,dilation_rate=1):
#    x = Conv2D(filters, size, strides=strides, padding=padding,dilation_rate=dilation_rate)(x)
#    x = ConvOffset2D(filters)(x)
    x = Conv2D(filters, size, strides=strides, padding=padding,dilation_rate=dilation_rate)(x)
    if activation == True:
        x = BatchActivate(x)
    return x

def fsBlock(blockInput, num_filters=16, batch_activate = False):
    x = ConvBlock(blockInput,num_filters,(3,3))
    
    supervision = Conv2D(1,(1,1),strides=(1,1),padding='same',activation='softmax')(x)
    
    attn_x = multiply([x,supervision])
    
    x_add = add([attn_x,x])
    
    x1 = ConvBlock(x_add,num_filters,(3,3))
    
    supervision1 = Conv2D(1,(1,1),strides=(1,1),padding='same',activation='softmax')(x1)
    
    attn_x1 = multiply([x1,supervision1])
    
    x1_add = add([attn_x1,x1])
    
    return x1_add

def residual_block(blockInput, num_filters=16, batch_activate = False,dilation_rate=1):
    x = BatchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) ,dilation_rate=dilation_rate)
    x = convolution_block(x, num_filters, (3,3), activation=False,dilation_rate=dilation_rate)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x




def bottleneck(x, filters_bottleneck=64, mode='cascade', depth=2,
               kernel_size=(3, 3), activation='relu'):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv2D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers) 

## register this optimizer to the global custom objects when it is imported
#get_custom_objects().update({'NormalizedOptimizer': NormalizedOptimizer})
#sgd = SGD(0.01, momentum=0.9, nesterov=True)
#sgd = NormalizedOptimizer(sgd, normalization='l2')
class SegmentionModel(ModelBase):
	def __init__(self,config=None):
		super(SegmentionModel, self).__init__(config)

		self.patch_height=config.patch_height
		self.patch_width = config.patch_width
		self.num_seg_class=config.seg_num
		self.multi_proportion = config.multi_proportion
#		self.build_model_resnet_4d()
#		self.build_model_casade_DeepAttention()
#		self.build_model_DeepAttention_3d()
#		self.build_model_DeepAttention()
#		self.build_model_resnet_3d()
#		self.build_Unet()
#		self.build_model_Dresnet_3d()
#		self.build_DenseConcatenateUnet()
#		self.build_model_DalitionUnet()
		self.build_model_DalitionUnet_binary()
#		self.build_SimpleUnet()
		self.save()
        
        

        
	def squeeze_excite_block_cSE(self,input, ratio=2):
		init = input
    
		filters = K.int_shape(init)[-1]
		se_shape = (1, 1, filters)
    
		se = GlobalAveragePooling2D()(init)
		se = Reshape(se_shape)(se)

		se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)(se)
		se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(se)
		x = multiply([init, se])
		return x
    
	def squeeze_excite_block_sSE(self,input):
		sSE_scale = Conv2D(1, (1, 1), activation='sigmoid', padding="same", use_bias = True)(input)
		return multiply([input, sSE_scale])
    
	def SEBlock(self,inputs,ratio):
		sSEx = self.squeeze_excite_block_sSE(inputs)
		cSEx = self.squeeze_excite_block_cSE(inputs,ratio) #modified 10/10/2018
		x = add(([sSEx, cSEx]))   
        
		return x
	def expend_as(self,tensor, rep):
		my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
		return my_repeat
	def AttnGatingBlock(self,x, g, inter_shape):
		shape_x = K.int_shape(x)  # 32
		shape_g = K.int_shape(g)  # 16

		theta_x = Conv2D(inter_shape, (3, 3), strides=(2, 2), padding='same')(x)  # 16
		shape_theta_x = K.int_shape(theta_x)

		phi_g = Conv2D(inter_shape, (1, 1), padding='same')(g)
		upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

		concat_xg = add([upsample_g, theta_x])
		act_xg = Activation('relu')(concat_xg)
		psi = Conv2D(1, (1, 1), padding='same')(act_xg)
		sigmoid_xg = Activation('sigmoid')(psi)
		shape_sigmoid = K.int_shape(sigmoid_xg)
		upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

		# my_repeat=Lambda(lambda xinput:K.repeat_elements(xinput[0],shape_x[1],axis=1))
		# upsample_psi=my_repeat([upsample_psi])
		upsample_psi = self.expend_as(upsample_psi, shape_x[3])

		y = multiply([upsample_psi, x])

		# print(K.is_keras_tensor(upsample_psi))

		result = Conv2D(shape_x[3], (3, 3), padding='same')(y)
		result_bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(result)
		return result_bn



	def DenseDilatedBlock(self,inputs, outdim):

		inputshape = K.int_shape(inputs)
		bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(inputs)
		act = Activation('relu')(bn)
		conv1 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)

		if inputshape[3] != outdim:
			shortcut = Conv2D(outdim, (1, 1), padding='same',dilation_rate=2)(inputs)
		else:
			shortcut = inputs
		result1 = add([conv1, shortcut])

		bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(result1)
		act = Activation('relu')(bn)
		conv2 = Conv2D(outdim, (3, 3), activation=None, padding='same',dilation_rate=4)(act)
		result = add([result1, conv2, shortcut])
		result = Activation('relu')(result)
#        result = multiply([inputs,result])
		return result



	def DenseBlock(self,inputs, outdim):

		inputshape = K.int_shape(inputs)
		bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(inputs)
		act = Activation('relu')(bn)
		conv1 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)

		if inputshape[3] != outdim:
			shortcut = Conv2D(outdim, (1, 1), padding='same')(inputs)
		else:
			shortcut = inputs
		result1 = add([conv1, shortcut])

		bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(result1)
		act = Activation('relu')(bn)
		conv2 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)
		result = add([result1, conv2, shortcut])
		result = Activation('relu')(result)
		result = Conv2D(outdim,(1,1),padding='same')(result)
		result = Activation('sigmoid')(result)
#		result = self.SEBlock(inputs=result,ratio=2)
		return result
    
	def build_SimpleUnet(self):
        
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = ConvBlock(inputs,start_neurons * 1 ,(3,3))
		conv1 = ConvBlock(conv1,start_neurons * 1 ,(3,3))        
		pool1 = MaxPooling2D((2, 2))(conv1)
        
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = ConvBlock(pool1,start_neurons * 2,(3,3))
		conv2 = ConvBlock(conv2,start_neurons * 2,(3,3))        
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
#		conv3 = ConvBlock(pool2,start_neurons * 4 ,(3,3))
#		conv3 = ConvBlock(conv3,start_neurons * 4 ,(3,3))        
#		pool3 = MaxPooling2D((2, 2))(conv3)

#		conv4 = ConvBlock(pool3,start_neurons * 8 ,(3,3))
#		conv4 = ConvBlock(conv4,start_neurons * 8 ,(3,3))        
#		pool4 = MaxPooling2D((2, 2))(conv4)    
        # 12 -> 6

#		pool4 = Dropout(0.5)(pool4)
#		MLF = concatenate([MaxPooling2D((4,4))(conv1),MaxPooling2D((2,2))(conv2),conv3])
#		MLF = Conv2D(1,(3,3),padding='same')(MLF)
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
        # Middle
#		convm = bottleneck(pool4)
		convm = ConvBlock(pool2,start_neurons * 4 ,(3,3))
		convm = ConvBlock(convm,start_neurons * 4 ,(3,3))         
		convm = ConvBlock(convm,start_neurons * 4 ,(3,3))         

#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12
#		deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)   
##		attn_3 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
##		attn_3 = Activation('softmax')(attn_3)
##		attn_3 = multiply([attn_3,conv3])
##		attn_3 = add([attn_3,conv3])        
#		uconv4 = concatenate([deconv4, conv4])   
#		uconv4 = ConvBlock(uconv4,start_neurons * 8 ,(3,3))
#		uconv4 = ConvBlock(uconv4,start_neurons * 8 ,(3,3)) #
 
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
#		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)   
##		attn_3 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
##		attn_3 = Activation('softmax')(attn_3)
##		attn_3 = multiply([attn_3,conv3])
##		attn_3 = add([attn_3,conv3])        
#		uconv3 = concatenate([deconv3, conv3])   
#		uconv3 = ConvBlock(uconv3,start_neurons * 4 ,(3,3))
#		uconv3 = ConvBlock(uconv3,start_neurons * 4 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     
        # 25 -> 50
        
		deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(convm)
#		attn_2 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_2 = Activation('softmax')(attn_2)
#		attn_2 = UpSampling2D(2)(attn_2)
#		attn_2 = multiply([attn_2,conv2])
#		attn_2 = add([attn_2,conv2])        
		uconv2 = concatenate([deconv2, conv2])
		uconv2 = ConvBlock(uconv2,start_neurons * 2 ,(3,3))
		uconv2 = ConvBlock(uconv2,start_neurons * 2 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     
#		uconv2 = self.SEBlock(uconv2,ratio=2)        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#		attn_1 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_1 = Activation('softmax')(attn_1)
#		attn_1 = UpSampling2D(4)(attn_1)
#		attn_1 = multiply([attn_1,conv1])
#		attn_1 = add([attn_1,conv1]) 
		uconv1 = concatenate([deconv1,conv1])
		uconv1 = ConvBlock(uconv1,start_neurons * 1 ,(3,3))
		uconv1 = ConvBlock(uconv1,start_neurons * 1 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     

#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model
    
	def build_model_DalitionUnet(self):
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 2)
		conv1 = residual_block(conv1,start_neurons * 2, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 4)
		conv2 = residual_block(conv2,start_neurons * 4, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
#		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
#		conv3 = residual_block(conv3,start_neurons * 4)
#		conv3 = residual_block(conv3,start_neurons * 4, True)
#		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    
        # 12 -> 6

#		pool4 = Dropout(0.5)(pool4)
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
    
        # Middle
#		convm = bottleneck(pool4)
		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		convm = residual_block(convm,start_neurons * 4)
		convm = residual_block(convm,start_neurons * 4, True)
        
#		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",dilation_rate=2)(convm)
#		convm = residual_block(convm,start_neurons * 4,dilation_rate=2)
#		convm = residual_block(convm,start_neurons * 4, True,dilation_rate=2)    
        
#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12

        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
#		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
#		uconv3 = concatenate([deconv3, conv3])    
##		uconv3 = Dropout(0.5)(uconv3)
#        
#		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
#		uconv3 = residual_block(uconv3,start_neurons * 4)
#		uconv3 = residual_block(uconv3,start_neurons * 4, True)
    
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv2 = concatenate([deconv2, conv2])
            
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 4)
		uconv2 = residual_block(uconv2,start_neurons * 4, True)
        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, conv1])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 2)
		uconv1 = residual_block(uconv1,start_neurons * 2, True)

#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

#		uconv0 = Conv2D(start_neurons , (3, 3), activation=None, padding="same")(hypercolumn)
#		uconv0 = residual_block(uconv0,start_neurons )
#		uconv0 = residual_block(uconv0,start_neurons , True)
        
		conv8 = Conv2D(self.num_seg_class+1, (1, 1), activation='relu', padding='same')(uconv1)
		act = Activation('softmax')(conv8)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
#        print()
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(act)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
#		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=conv8)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        
#		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		#model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model  
    
    
    
	def build_model_DalitionUnet_binary(self):
        # 101 -> 50
		if self.multi_proportion:
		     print('Model Input Channel : 1')            
		     inputs = Input((self.patch_height, self.patch_width, 1))
		else:
		     print('Model Input Channel : 3')             
		     inputs = Input((self.patch_height, self.patch_width, 3))            
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 2)
		conv1 = residual_block(conv1,start_neurons * 2, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 4)
		conv2 = residual_block(conv2,start_neurons * 4, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
#		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
#		conv3 = residual_block(conv3,start_neurons * 4)
#		conv3 = residual_block(conv3,start_neurons * 4, True)
#		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    
        # 12 -> 6

#		pool4 = Dropout(0.5)(pool4)
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
    
        # Middle
#		convm = bottleneck(pool4)
		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		convm = residual_block(convm,start_neurons * 4)
		convm = residual_block(convm,start_neurons * 4, True)
        
#		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",dilation_rate=2)(convm)
#		convm = residual_block(convm,start_neurons * 4,dilation_rate=2)
#		convm = residual_block(convm,start_neurons * 4, True,dilation_rate=2)    
        
#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12

        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
#		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
#		uconv3 = concatenate([deconv3, conv3])    
##		uconv3 = Dropout(0.5)(uconv3)
#        
#		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
#		uconv3 = residual_block(uconv3,start_neurons * 4)
#		uconv3 = residual_block(uconv3,start_neurons * 4, True)
    
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv2 = concatenate([deconv2, conv2])
            
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 4)
		uconv2 = residual_block(uconv2,start_neurons * 4, True)
        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, conv1])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 2)
		uconv1 = residual_block(uconv1,start_neurons * 2, True)

#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

#		uconv0 = Conv2D(start_neurons , (3, 3), activation=None, padding="same")(hypercolumn)
#		uconv0 = residual_block(uconv0,start_neurons )
#		uconv0 = residual_block(uconv0,start_neurons , True)
        
		conv8 = Conv2D(self.num_seg_class, (1, 1), activation=None, padding='same')(uconv1)
		act = Activation('sigmoid')(conv8)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
#		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
#		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss=dice_loss, metrics=['binary_accuracy',dice_coef])
        
#		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		#model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model      
    
    
	def build_model_se_resnet_4d(self):
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 2)
		conv1 = residual_block(conv1,start_neurons * 2, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 4)
		conv2 = residual_block(conv2,start_neurons * 4, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		conv3 = residual_block(conv3,start_neurons * 4)
		conv3 = residual_block(conv3,start_neurons * 4, True)
		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    
        # 12 -> 6
		conv4 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool3)
		conv4 = residual_block(conv4,start_neurons * 4)
		conv4 = residual_block(conv4,start_neurons * 4, True)
		pool4 = MaxPooling2D((2, 2))(conv4)
#		pool4 = Dropout(0.5)(pool4)
		
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
        # Middle
#		convm = bottleneck(pool4)
		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool4)
		convm = residual_block(convm,start_neurons * 4)
		convm = residual_block(convm,start_neurons * 4, True)
        
#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12
		deconv4 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv4 = concatenate([deconv4, conv4])
#		uconv4 = Dropout(0.5)(uconv4)
        
		uconv4 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv4)
		uconv4 = residual_block(uconv4,start_neurons * 4)
		uconv4 = residual_block(uconv4,start_neurons * 4, True)
		uconv4 = self.SEBlock(uconv4,ratio=2)        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		uconv3 = concatenate([deconv3, conv3])    
#		uconv3 = Dropout(0.5)(uconv3)
        
		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
		uconv3 = residual_block(uconv3,start_neurons * 4)
		uconv3 = residual_block(uconv3,start_neurons * 4, True)
		uconv3 = self.SEBlock(uconv3,ratio=2)     
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
		uconv2 = concatenate([deconv2, conv2])
            
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 4)
		uconv2 = residual_block(uconv2,start_neurons * 4, True)
		uconv2 = self.SEBlock(uconv2,ratio=2)        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, conv1])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 2)
		uconv1 = residual_block(uconv1,start_neurons * 2, True)
		uconv1 = self.SEBlock(uconv1,ratio=2)  
#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model

	def build_Unet(self):
        
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = ConvBlock(inputs,start_neurons * 1 ,(3,3))
		conv1 = ConvBlock(conv1,start_neurons * 1 ,(3,3))        
		pool1 = MaxPooling2D((2, 2))(conv1)
        
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = ConvBlock(pool1,start_neurons * 2,(3,3))
		conv2 = ConvBlock(conv2,start_neurons * 2,(3,3))        
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3 = ConvBlock(pool2,start_neurons * 4 ,(3,3))
		conv3 = ConvBlock(conv3,start_neurons * 4 ,(3,3))        
		pool3 = MaxPooling2D((2, 2))(conv3)

		conv4 = ConvBlock(pool3,start_neurons * 8 ,(3,3))
		conv4 = ConvBlock(conv4,start_neurons * 8 ,(3,3))        
		pool4 = MaxPooling2D((2, 2))(conv4)    
        # 12 -> 6

#		pool4 = Dropout(0.5)(pool4)
#		MLF = concatenate([MaxPooling2D((4,4))(conv1),MaxPooling2D((2,2))(conv2),conv3])
#		MLF = Conv2D(1,(3,3),padding='same')(MLF)
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
        # Middle
#		convm = bottleneck(pool4)
		convm = ConvBlock(pool4,start_neurons * 16 ,(3,3))
		convm = ConvBlock(convm,start_neurons * 16 ,(3,3))         
		convm = ConvBlock(convm,start_neurons * 16 ,(3,3))         

#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12
		deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)   
#		attn_3 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_3 = Activation('softmax')(attn_3)
#		attn_3 = multiply([attn_3,conv3])
#		attn_3 = add([attn_3,conv3])        
		uconv4 = concatenate([deconv4, conv4])   
		uconv4 = ConvBlock(uconv4,start_neurons * 8 ,(3,3))
		uconv4 = ConvBlock(uconv4,start_neurons * 8 ,(3,3)) #
 
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)   
#		attn_3 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_3 = Activation('softmax')(attn_3)
#		attn_3 = multiply([attn_3,conv3])
#		attn_3 = add([attn_3,conv3])        
		uconv3 = concatenate([deconv3, conv3])   
		uconv3 = ConvBlock(uconv3,start_neurons * 4 ,(3,3))
		uconv3 = ConvBlock(uconv3,start_neurons * 4 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     
        # 25 -> 50
        
		deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
#		attn_2 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_2 = Activation('softmax')(attn_2)
#		attn_2 = UpSampling2D(2)(attn_2)
#		attn_2 = multiply([attn_2,conv2])
#		attn_2 = add([attn_2,conv2])        
		uconv2 = concatenate([deconv2, conv2])
		uconv2 = ConvBlock(uconv2,start_neurons * 2 ,(3,3))
		uconv2 = ConvBlock(uconv2,start_neurons * 2 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     
#		uconv2 = self.SEBlock(uconv2,ratio=2)        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#		attn_1 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_1 = Activation('softmax')(attn_1)
#		attn_1 = UpSampling2D(4)(attn_1)
#		attn_1 = multiply([attn_1,conv1])
#		attn_1 = add([attn_1,conv1]) 
		uconv1 = concatenate([deconv1,conv1])
		uconv1 = ConvBlock(uconv1,start_neurons * 1 ,(3,3))
		uconv1 = ConvBlock(uconv1,start_neurons * 1 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     

#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model
        
	def build_DenseConcatenateUnet(self):
        
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1_0 = ConvBlock(inputs,start_neurons * 1 ,(3,3))
		conv1_1 = ConvBlock(conv1_0,start_neurons * 1 ,(3,3))        
		pool1 = MaxPooling2D((2, 2))(conv1_1)
        
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2_0 = ConvBlock(pool1,start_neurons * 2,(3,3))
		conv2_1 = ConvBlock(conv2_0,start_neurons * 2,(3,3))        
		pool2 = MaxPooling2D((2, 2))(conv2_1)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3_0 = ConvBlock(pool2,start_neurons * 4 ,(3,3))
		conv3_1 = ConvBlock(conv3_0,start_neurons * 4 ,(3,3))        
		pool3 = MaxPooling2D((2, 2))(conv3_1)
    
        # 12 -> 6

#		pool4 = Dropout(0.5)(pool4)
#		MLF = concatenate([MaxPooling2D((4,4))(conv1),MaxPooling2D((2,2))(conv2),conv3])
#		MLF = Conv2D(1,(3,3),padding='same')(MLF)
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
        # Middle
#		convm = bottleneck(pool4)
		convm_0 = ConvBlock(pool3,start_neurons * 8 ,(3,3))
		convm_1 = ConvBlock(convm_0,start_neurons * 8 ,(3,3))         

#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12
 
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm_1)   
#		attn_3 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_3 = Activation('softmax')(attn_3)
#		attn_3 = multiply([attn_3,conv3])
#		attn_3 = add([attn_3,conv3])        
		uconv3 = concatenate([deconv3, conv3_1])   
		uconv3 = ConvBlock(uconv3,start_neurons * 4 ,(3,3))
		uconv3 = concatenate([uconv3,conv3_0])
		uconv3 = ConvBlock(uconv3,start_neurons * 4 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     
        # 25 -> 50
        
		deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
#		attn_2 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_2 = Activation('softmax')(attn_2)
#		attn_2 = UpSampling2D(2)(attn_2)
#		attn_2 = multiply([attn_2,conv2])
#		attn_2 = add([attn_2,conv2])        
		uconv2 = concatenate([deconv2, conv2_1])
		uconv2 = ConvBlock(uconv2,start_neurons * 2 ,(3,3))
		uconv2 = concatenate([uconv2,conv2_0] )
		uconv2 = ConvBlock(uconv2,start_neurons * 2 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     
#		uconv2 = self.SEBlock(uconv2,ratio=2)        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
#		attn_1 = Conv2D(1,(3,3),activation=None,padding='same')(MLF)
#		attn_1 = Activation('softmax')(attn_1)
#		attn_1 = UpSampling2D(4)(attn_1)
#		attn_1 = multiply([attn_1,conv1])
#		attn_1 = add([attn_1,conv1]) 
		uconv1 = concatenate([deconv1, conv1_1])
		uconv1 = ConvBlock(uconv1,start_neurons * 1 ,(3,3))
		uconv1 = concatenate([uconv1,  conv1_0])
		uconv1 = ConvBlock(uconv1,start_neurons * 1 ,(3,3)) #		uconv3 = self.SEBlock(uconv3,ratio=2)     

#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model

	def build_model_Dresnet_3d(self):
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 1)
#		conv1 = residual_block(conv1,start_neurons * 1, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 2)
#		conv2 = residual_block(conv2,start_neurons * 2, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		conv3 = residual_block(conv3,start_neurons * 4)
#		conv3 = residual_block(conv3,start_neurons * 4, True)
		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    
        # 12 -> 6
#		pool4 = Dropout(0.5)(pool4)
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
    
        # Middle
#		convm = bottleneck(pool4)
		convm = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
		convm = residual_block(convm,start_neurons * 8)
#		convm = residual_block(convm,start_neurons * 8, True)
        
#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        

        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv3 = concatenate([deconv3, conv3])    
#		uconv3 = Dropout(0.5)(uconv3)
        
		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
		uconv3 = residual_block(uconv3,start_neurons * 4)
#		uconv3 = residual_block(uconv3,start_neurons * 4, True)
    
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
		uconv2 = concatenate([deconv2, conv2])
            
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 2)
#		uconv2 = residual_block(uconv2,start_neurons * 2, True)
        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, conv1])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 1)
#		uconv1 = residual_block(uconv1,start_neurons * 1, True)

#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

#		uconv0 = Conv2D(start_neurons , (3, 3), activation=None, padding="same")(hypercolumn)
#		uconv0 = residual_block(uconv0,start_neurons )
#		uconv0 = residual_block(uconv0,start_neurons , True)
        
		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model


	def build_model_resnet_3d(self):
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 2)
		conv1 = residual_block(conv1,start_neurons * 2, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 4)
		conv2 = residual_block(conv2,start_neurons * 4, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		conv3 = residual_block(conv3,start_neurons * 4)
		conv3 = residual_block(conv3,start_neurons * 4, True)
		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    
        # 12 -> 6

#		pool4 = Dropout(0.5)(pool4)
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
    
        # Middle
#		convm = bottleneck(pool4)
		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool3)
		convm = residual_block(convm,start_neurons * 4)
		convm = residual_block(convm,start_neurons * 4, True)
        
#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12

        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv3 = concatenate([deconv3, conv3])    
#		uconv3 = Dropout(0.5)(uconv3)
        
		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
		uconv3 = residual_block(uconv3,start_neurons * 4)
		uconv3 = residual_block(uconv3,start_neurons * 4, True)
    
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
		uconv2 = concatenate([deconv2, conv2])
            
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 4)
		uconv2 = residual_block(uconv2,start_neurons * 4, True)
        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, conv1])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 2)
		uconv1 = residual_block(uconv1,start_neurons * 2, True)

#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

#		uconv0 = Conv2D(start_neurons , (3, 3), activation=None, padding="same")(hypercolumn)
#		uconv0 = residual_block(uconv0,start_neurons )
#		uconv0 = residual_block(uconv0,start_neurons , True)
        
		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model  

	def build_model_resnet_4d(self):
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 2)
		conv1 = residual_block(conv1,start_neurons * 2, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 4)
		conv2 = residual_block(conv2,start_neurons * 4, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		conv3 = residual_block(conv3,start_neurons * 4)
		conv3 = residual_block(conv3,start_neurons * 4, True)
		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    
        # 12 -> 6
		conv4 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool3)
		conv4 = residual_block(conv4,start_neurons * 4)
		conv4 = residual_block(conv4,start_neurons * 4, True)
		pool4 = MaxPooling2D((2, 2))(conv4)
#		pool4 = Dropout(0.5)(pool4)
#		MLF = concatenate([MaxPooling2D((16,16))(conv1),MaxPooling2D((8,8))(conv2),MaxPooling2D((4,4))(conv3),pool4])
    
        # Middle
#		convm = bottleneck(pool4)
		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool4)
		convm = residual_block(convm,start_neurons * 4)
		convm = residual_block(convm,start_neurons * 4, True)
        
#		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12
		deconv4 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv4 = concatenate([deconv4, conv4])
#		uconv4 = Dropout(0.5)(uconv4)
        
		uconv4 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv4)
		uconv4 = residual_block(uconv4,start_neurons * 4)
		uconv4 = residual_block(uconv4,start_neurons * 4, True)
        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		uconv3 = concatenate([deconv3, conv3])    
#		uconv3 = Dropout(0.5)(uconv3)
        
		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
		uconv3 = residual_block(uconv3,start_neurons * 4)
		uconv3 = residual_block(uconv3,start_neurons * 4, True)
    
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
		uconv2 = concatenate([deconv2, conv2])
            
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 4)
		uconv2 = residual_block(uconv2,start_neurons * 4, True)
        
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, conv1])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 2)
		uconv1 = residual_block(uconv1,start_neurons * 2, True)

#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)

#		uconv0 = Conv2D(start_neurons , (3, 3), activation=None, padding="same")(hypercolumn)
#		uconv0 = residual_block(uconv0,start_neurons )
#		uconv0 = residual_block(uconv0,start_neurons , True)
        
		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		model.summary()
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model   
        
	def build_model(self):
		inputs = Input((self.patch_height, self.patch_width, 1))
		conv1 = Conv2D(32, (1, 1), activation=None, padding='same')(inputs)
		conv1 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv1)
		conv1 = Activation('relu')(conv1)

		conv1 = self.DenseBlock(conv1, 32)  # 48
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = self.DenseBlock(pool1, 64)  # 24
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = self.DenseBlock(pool2, 64)  # 12
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = self.DenseBlock(pool3, 64)  # 12

		up1 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv4)
		up1 = concatenate([up1, conv3], axis=3)

		conv5 = self.DenseBlock(up1, 64)

		up2 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv5)
		up2 = concatenate([up2, conv2], axis=3)

		conv6 = self.DenseBlock(up2, 64)

		up3 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(conv6)
		up3 = concatenate([up3, conv1], axis=3)

		conv7 = self.DenseBlock(up3, 32)
#		conv7 = Dropout(0.5)(conv7)
		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(conv7)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)
		# for theano
		#conv8 = core.Reshape(((self.num_seg_class + 1), self.patch_height * self.patch_width))(conv8)
		#conv8 = core.Permute((2, 1))(conv8)
		############
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model     
        
	def build_model_DeepAttention(self):
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 2)
		conv1 = residual_block(conv1,start_neurons * 2, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 4)
		conv2 = residual_block(conv2,start_neurons * 4, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		conv3 = residual_block(conv3,start_neurons * 4)
		conv3 = residual_block(conv3,start_neurons * 4, True)
		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    
        # 12 -> 6
		conv4 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool3)
		conv4 = residual_block(conv4,start_neurons * 4)
		conv4 = residual_block(conv4,start_neurons * 4, True)
		pool4 = MaxPooling2D((2, 2))(conv4)
#		pool4 = Dropout(0.5)(pool4)
    
        # Middle
		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool4)
#		convm = bottleneck(pool4)
		convm = residual_block(convm,start_neurons * 4)
		convm = residual_block(convm,start_neurons * 4, True)
        
		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12
		deconv4 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv4 = concatenate([deconv4, attn_1])
#		uconv4 = Dropout(0.5)(uconv4)
        
		uconv4 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv4)
		uconv4 = residual_block(uconv4,start_neurons * 4)
		uconv4 = residual_block(uconv4,start_neurons * 4, True)

		attn_2 = self.AttnGatingBlock(conv3, uconv4, start_neurons * 4)        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		uconv3 = concatenate([deconv3,attn_2])    
#		uconv3 = Dropout(0.5)(uconv3)
        
		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
		uconv3 = residual_block(uconv3,start_neurons * 4)
		uconv3 = residual_block(uconv3,start_neurons * 4, True)

		attn_3 = self.AttnGatingBlock(conv2, uconv3, start_neurons * 4)       
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
		uconv2 = concatenate([deconv2, attn_3])
     
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 4)
		uconv2 = residual_block(uconv2,start_neurons * 4, True)
        
		attn_4 = self.AttnGatingBlock(conv1, uconv2, start_neurons * 2)         
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, attn_4])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 2)
		uconv1 = residual_block(uconv1,start_neurons * 2, True)


#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)
		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model     

	def build_model_DeepAttention_3d(self):
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 2)
		conv1 = residual_block(conv1,start_neurons * 2, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 4)
		conv2 = residual_block(conv2,start_neurons * 4, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		conv3 = residual_block(conv3,start_neurons * 4)
		conv3 = residual_block(conv3,start_neurons * 4, True)
		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    

    
        # Middle
#		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool3)
		convm = bottleneck(pool3)
		convm = residual_block(convm,start_neurons * 4)
		convm = residual_block(convm,start_neurons * 4, True)
        

		attn_1 = self.AttnGatingBlock(conv3, convm, start_neurons * 4)        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv3 = concatenate([deconv3,attn_1])    
#		uconv3 = Dropout(0.5)(uconv3)
        
		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
		uconv3 = residual_block(uconv3,start_neurons * 4)
		uconv3 = residual_block(uconv3,start_neurons * 4, True)

		attn_3 = self.AttnGatingBlock(conv2, uconv3, start_neurons * 4)       
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
		uconv2 = concatenate([deconv2, attn_3])
     
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 4)
		uconv2 = residual_block(uconv2,start_neurons * 4, True)
        
		attn_4 = self.AttnGatingBlock(conv1, uconv2, start_neurons * 2)         
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, attn_4])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 2)
		uconv1 = residual_block(uconv1,start_neurons * 2, True)


#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3)],-1)
		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		model.summary()
		self.model = model 


	def build_model_DeepAttention(self):
        # 101 -> 50
		inputs = Input((self.patch_height, self.patch_width, 1))
		start_neurons = 16
		conv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(inputs)
		conv1 = residual_block(conv1,start_neurons * 2)
		conv1 = residual_block(conv1,start_neurons * 2, True)
		pool1 = MaxPooling2D((2, 2))(conv1)
#		pool1 = Dropout(0.25)(pool1)
    
        # 50 -> 25
		conv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool1)
		conv2 = residual_block(conv2,start_neurons * 4)
		conv2 = residual_block(conv2,start_neurons * 4, True)
		pool2 = MaxPooling2D((2, 2))(conv2)
#		pool2 = Dropout(0.5)(pool2)
    
        # 25 -> 12
		conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
		conv3 = residual_block(conv3,start_neurons * 4)
		conv3 = residual_block(conv3,start_neurons * 4, True)
		pool3 = MaxPooling2D((2, 2))(conv3)
#		pool3 = Dropout(0.5)(pool3)
    
        # 12 -> 6
		conv4 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool3)
		conv4 = residual_block(conv4,start_neurons * 4)
		conv4 = residual_block(conv4,start_neurons * 4, True)
		pool4 = MaxPooling2D((2, 2))(conv4)
#		pool4 = Dropout(0.5)(pool4)
    
        # Middle
		convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool4)
#		convm = bottleneck(pool4)
		convm = residual_block(convm,start_neurons * 4)
		convm = residual_block(convm,start_neurons * 4, True)
        
		attn_1 = self.AttnGatingBlock(conv4, convm, start_neurons * 4)        
        # 6 -> 12
		deconv4 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
		uconv4 = concatenate([deconv4, attn_1])
#		uconv4 = Dropout(0.5)(uconv4)
        
		uconv4 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv4)
		uconv4 = residual_block(uconv4,start_neurons * 4)
		uconv4 = residual_block(uconv4,start_neurons * 4, True)

		attn_2 = self.AttnGatingBlock(conv3, uconv4, start_neurons * 4)        
        # 12 -> 25
        #deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
		uconv3 = concatenate([deconv3,attn_2])    
#		uconv3 = Dropout(0.5)(uconv3)
        
		uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
		uconv3 = residual_block(uconv3,start_neurons * 4)
		uconv3 = residual_block(uconv3,start_neurons * 4, True)

		attn_3 = self.AttnGatingBlock(conv2, uconv3, start_neurons * 4)       
        # 25 -> 50
		deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
		uconv2 = concatenate([deconv2, attn_3])
     
#		uconv2 = Dropout(0.5)(uconv2)
		uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
		uconv2 = residual_block(uconv2,start_neurons * 4)
		uconv2 = residual_block(uconv2,start_neurons * 4, True)
        
		attn_4 = self.AttnGatingBlock(conv1, uconv2, start_neurons * 2)         
        # 50 -> 101
        #deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
		deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
		uconv1 = concatenate([deconv1, attn_4])
        
#		uconv1 = Dropout(0.5)(uconv1)
		uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
		uconv1 = residual_block(uconv1,start_neurons * 2)
		uconv1 = residual_block(uconv1,start_neurons * 2, True)


#		hypercolumn = concatenate([uconv1,UpSampling2D(2)(uconv2),UpSampling2D(4)(uconv3),UpSampling2D(8)(uconv4)],-1)
		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(uconv1)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)        
        #uconv1 = Dropout(DropoutRatio/2)(uconv1)
        #output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		#plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model     


class OptimizerWrapper(optimizers.Optimizer):

    def __init__(self, optimizer):     
        
        self.optimizer = optimizers.get(optimizer)

        # patch the `get_gradients` call
        self._optimizer_get_gradients = self.optimizer.get_gradients

    def get_gradients(self, loss, params):      
        grads = self._optimizer_get_gradients(loss, params)
        return grads

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        # monkey patch `get_gradients`
        self.optimizer.get_gradients = self.get_gradients

        # get the updates
        self.optimizer.get_updates(loss, params)

        # undo monkey patch
        self.optimizer.get_gradients = self._optimizer_get_gradients

        return self.updates

    def set_weights(self, weights):       
        self.optimizer.set_weights(weights)

    def get_weights(self):        
        return self.optimizer.get_weights()

    def get_config(self):       
        # properties of NormalizedOptimizer
        config = {'optimizer_name': self.optimizer.__class__.__name__.lower()}

        # optimizer config
        optimizer_config = {'optimizer_config': self.optimizer.get_config()}
        return dict(list(optimizer_config.items()) + list(config.items()))

    @property
    def weights(self):
        return self.optimizer.weights

    @property
    def updates(self):
        return self.optimizer.updates

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    @classmethod
    def set_normalization_function(cls, name, func):
        global _NORMS
        _NORMS[name] = func

    @classmethod
    def get_normalization_functions(cls):        
        global _NORMS
        return sorted(list(_NORMS.keys()))


class NormalizedOptimizer(OptimizerWrapper):

    def __init__(self, optimizer, normalization='l2'):       
        super(NormalizedOptimizer, self).__init__(optimizer)

        if normalization not in _NORMS:
            raise ValueError('`normalization` must be one of %s.\n' 
                             'Provided was "%s".' % (str(sorted(list(_NORMS.keys()))), normalization))

        self.normalization = normalization
        self.normalization_fn = _NORMS[normalization]
        self.lr = K.variable(1e-3, name='lr')

    def get_gradients(self, loss, params):       
        grads = super(NormalizedOptimizer, self).get_gradients(loss, params)
        grads = [grad / self.normalization_fn(grad) for grad in grads]
        return grads

    def get_config(self):        
        # properties of NormalizedOptimizer
        config = {'normalization': self.normalization}

        # optimizer config
        base_config = super(NormalizedOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):       
        optimizer_config = {'class_name': config['optimizer_name'],
                            'config': config['optimizer_config']}

        optimizer = optimizers.get(optimizer_config)
        normalization = config['normalization']

        return cls(optimizer, normalization=normalization)


_NORMS = {
    'l2': l2_norm,
}


        
#        return output_layer
#config = {
#  "exp_name": "VesselNet",
#  "epochs": 20,
#  "batch_size": 25,
#  "patch_height": 96,
#  "patch_width": 96,
#  "subsample": 500,
#  "total_train": 40,
#  "total_val": 20,
#  "train_datatype": "tif",
#  "train_gt_datatype": "tif",
#  "val_datatype": "tif",
#  "val_gt_datatype": "tif",
#  "test_datatype": "tif",
#  "test_gt_datatype": "tif",
#  "height": 584,
#  "width": 565,
#  "stride_height": 5,
#  "stride_width": 5,
#  "seg_num": 1
#}     
#model = SegmentionModel(config)