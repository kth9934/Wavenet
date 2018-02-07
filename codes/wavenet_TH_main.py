import wave
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation
from keras.layers import Conv1D, Dense, Activation, Merge

from taehyoung_util import *
from wavenet_TH_models import *

batch_size = 1
epoch = 700

##################################### 데이타세트 만들기 #######################################


wavefile = []

for i in range(len(wavfile)):                                    #concatenate 하기 위해서 차원을 늘려주는데,
    wavefile.append(wavfile[i].reshape(1, len(wavfile[0]), 256)) #변수를 새로 설정해줘야한다.(wavfile => wavefile)
    x_train = wavefile[0][:,:-1,:]                                   #x_train은 전체에서 마지막 column하나 뺀거
    y_train = wavefile[0][:,1:,:]                                    #y_train은 맨앞에 하나 뺀거

for i in range (len(wavfile)-1):                                        #concatenate 해준다.
    x_train = np.concatenate((x_train,wavefile[i+1][:,:-1,:]), axis=0)
    y_train = np.concatenate((y_train,wavefile[i+1][:,1:,:]), axis=0)


x_test = x_train
y_test = y_train


# print(x_train.shape)
# print(y_train.shape)

################################### TRAINING ############################################################

model.compile(optimizer='adam',  loss='categorical_crossentropy', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

loss = score[0]
accuracy = score[1]

print("Accuracy : ", accuracy, "loss : ", loss)

model_save = model.save('WaveNet.h5')
model_weight = model.save_weights('WaveNet_Weights.h5', overwrite=True)