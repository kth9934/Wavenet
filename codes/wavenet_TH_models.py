from __future__ import division

import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keras.models import Model, Sequential
from keras.layers import ( Input, Activation, Dense, Flatten, Lambda, Concatenate, Add, merge,)
from keras.layers.convolutional import ( Conv1D, Conv2D, MaxPooling2D, AveragePooling2D,
										 Conv3D, MaxPooling3D, AveragePooling3D )
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences as padseq
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences as padseq
from keras.engine.topology import Layer

from taehyoung_util import *
vals = []
dim = 256
path = "/home/kth99343733/wavenet/test"
wavfile = raw_audio(path,vals)            # len(wavfile) = 총 sample 개수 ,  #wavfile[i] = 한 샘플
wavfile = padseq(wavfile, padding = 'pre')
filters = 2
kernel_size = 2
timestep = wavfile.shape[1]-1

#####################BUILD MODEL########################################

inputs = Input(shape=(timestep,dim))

kout = (Conv1D(filters=30, kernel_size=2, activation='tanh',
               input_shape=(timestep, dim), padding = 'causal'))(inputs)

##################### BUILD LAYERS #########################################

Dial_conv_dict = { }
Dial_conv_out = { }
skip_out = { }
layer_oneby = { }

dd = kout

#for k in range (1, 4):   #기존에는 10개짜리 세트 3개

def tan_sig(x):                            #tanh랑 sigmoid 곱하기 연산 해주기 위한 함수
    t = Activation('tanh')(x[:,:,0])
    s = Activation('sigmoid')(x[:,:,1])    #dial_conv에서 나온 두개의 아웃풋을 각각 하나씩 tanh와 sigmoid에 넣음
    t = K.expand_dims(t,-1)
    s = K.expand_dims(s,-1)                #activation 과정에서 필터를 하나씩 뽑으면 차원이 줄어들기 때문에 다시 차원을 늘려줌.
    return t * s                           #tanh랑 sigmoid를 거치고 나온 값들을 곱해서 return함.

def res_convout(x,dd):                        #마지막 residual과 1by1_conv를 더해주기 위한 함수
    return x + dd                          #1by1_conv를 거치고 나온 아웃풋(x = out)과 dd(residual)을 더해서 return


#skip으로 뽑아낼때 1by1 거쳐서 나갈 수 있도록 해주는 함수
#def s_out(x):
#    return Conv1D(filters=1, kernel_size=1,use_bias=True,
#                  kernel_initializer='glorot_uniform', bias_initializer='zeros',
#                  kernel_regularizer='l2')(x)

dial_val = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,    # dialration rate들
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

for i in range(30):
    Dial_conv_dict["Dial_rate_{}".format(i)] = Conv1D(filters=60, kernel_size=2, padding='causal',
                                                      dilation_rate=dial_val[i], activation=None, use_bias=True,
                                                      kernel_initializer='glorot_uniform',
                                                      bias_initializer='zeros', kernel_regularizer='l2')(dd)

    Dial_conv_out["Dial_conv_out_{}".format(i)] = Dial_conv_dict["Dial_rate_{}".format(i)]  # 위에서 dial_conv거친 아웃풋

    dial_out = Dial_conv_out["Dial_conv_out_{}".format(i)]

    out = Lambda(tan_sig)(dial_out)  # tanh랑 sig 곱하는 함수 불러옴

    if i == 0:  # 첫번째 포문 돌때는 그냥 sout을 쓰고, 두번째부터는 뒤에다가 concatenate로 쌓음.
        skip_out = out  # 마지막에 output으로 for문 돌면서 쌓여왔던 skip_out을 한번에 뽑을 수 있음.
    elif i > 0:
        # skip_out = Concatenate(axis=-1)([skip_out, sout])
        skip_out = Concatenate(axis=-1)([skip_out, out])

    layer_oneby["layer_oneby_{}".format(i)] = Conv1D(filters=30, kernel_size=1, use_bias=True,  # 1by1 conv
                                                     kernel_initializer='glorot_uniform', bias_initializer='zeros',
                                                     kernel_regularizer='l2')(out)

    #layer_output = Lambda(res_convout)([layer_oneby["layer_oneby_{}".format(i)], dd])  # 1by1 conv랑 residual 더하는함수 불러옴
    layer_output = Add()([layer_oneby["layer_oneby_{}".format(i)], dd])  # 1by1 conv랑 residual 더하는함수 불러옴
    dd = layer_output  # layer_out을 다시 인풋으로 쓰기위해서 인풋을 layer_out으로 설정해줌

############################################################################################################

#####BUILD SKIP-CONNECTION#################################

f = Conv1D(filters=256, kernel_size=1, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer='l2')(skip_out)
f = Activation('relu')(f)
f = Conv1D(filters=256, kernel_size=1, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer='l2')(f)
f = Activation('relu')(f)
f = Conv1D(filters=256, kernel_size=1, use_bias=True,
           kernel_initializer='glorot_uniform', bias_initializer='zeros',
           kernel_regularizer='l2')(f)
output = Activation('softmax')(f)

model = Model(inputs = inputs, outputs = output)


# # model.summary()
# # model.predict(final_input)
# # print(model.predict(final_input).shape)
# #  현재는 r = 1 로 두었기 때문에, 아웃풋이 30개지만, 나중에 r 과 s를 조정해야한다.
