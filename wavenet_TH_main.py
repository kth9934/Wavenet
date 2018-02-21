import wave
import numpy as np
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Activation, Dropout
from keras.layers import Conv1D, Dense, Activation, Merge
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint
from taehyoung_util import *
from wavenet_TH_models import *
# import numpy as np



batch_size = 1
epoch = 11

##################################### 데이타세트 만들기 #######################################

#vctk 데이터셋 리스트 모두 불러오기
all_files = glob(os.path.join('../vctk/VCTK-Corpus/trainset_np', '*npy'))

# 데이터셋 초기 선언
x_train = np.load(all_files[0])
x_train = np.array(x_train[5000:15000], dtype='uint8')
x_train = np.array(mu_to_onehot(x_train), dtype='uint8')
x_train = x_train.reshape(1, 10000, 256)
y_train = np.load(all_files[0])
y_train = np.array(y_train[5001:15001], dtype='uint8')
y_train = np.array(mu_to_onehot(y_train), dtype='uint8')
y_train = y_train.reshape(1, 10000, 256)
x_val = np.load(all_files[10000])
x_val = np.array(x_val[5000:15000], dtype='uint8')
x_val = np.array(mu_to_onehot(x_val), dtype='uint8')
x_val = x_val.reshape(1, 10000, 256)
y_val = np.load(all_files[10000])
y_val = np.array(y_val[5001:15001], dtype='uint8')
y_val = np.array(mu_to_onehot(y_val), dtype='uint8')
y_val = y_val.reshape(1, 10000, 256)

#데이터셋 concatenate로 붙여서 하나로 만들기
for i in range (6000):
    a = np.load(all_files[i+1])
    a = a[5000:15001]
    a = np.array(mu_to_onehot(a), dtype='uint8')
    a = a.reshape(1, 10001, 256)
    x_train = np.concatenate((x_train, a[:,:-1,:]), axis=0)
    y_train = np.concatenate((y_train, a[:,1:,:]), axis=0)
    print(i,'th train set is completed, (',i/6000,'%)')

for i in range(500):
    b = np.load(all_files[9000+i+1])
    b = b[5000:15001]
    b = np.array(mu_to_onehot(b), dtype='uint8')
    b = b.reshape(1, 10001, 256)
    x_val = np.concatenate((x_val, b[:,:-1,:]), axis =0)
    y_val = np.concatenate((y_val, b[:,1:,:]), axis=0)
    print(i, 'th train set is completed, (',i/500,'%)')

#데이터셋 저장 ( 저장 후에는 저장된것만 불러와서 쓰면 됌)
np.save('x_train', x_train)
np.save('y_train', y_train)
np.save('x_val', x_val)
np.save('y_val', y_val)
# print(x_train.shape)
# print(y_train.shape)

################################### TRAINING ############################################################



callbacks = [EarlyStopping(monitor='val_loss',
                           patience=4,
                           min_delta=0.00001,
                           mode='min'),
            ReduceLROnPlateau(monitor='val_loss',
                               factor=0.8,
                               patience=3,
                               epsilon=0.00001,
                               mode='min')]


model.fit(x_train,y_train, batch_size=batch_size, epochs=epoch, verbose=1, callbacks=callbacks,
          validation_data= (x_val, y_val))

#score = model.evaluate(x_test, y_test, verbose=0)

#loss = score[0]
#accuracy = score[1]

#print("Accuracy : ", accuracy, "loss : ", loss)

model_save = model.save('WaveNet_TH.h5')
model_weight = model.save_weights('Weights_TH.h5', overwrite=True)