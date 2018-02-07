from keras.models import load_model
import numpy as np
import simplejson as sp

from taehyoung_util import load_to_onehot
import matplotlib.pyplot as plt
from wavenet_TH_models import *

timestep = 55332
model = load_model('WaveNet.h5')
model.load_weights('Weights_700.h5')

wavefile = []

for i in range(len(wavfile)):                                    #concatenate 하기 위해서 차원을 늘려주는데,
    wavefile.append(wavfile[i].reshape(1, len(wavfile[0]), 256)) #변수를 새로 설정해줘야한다.(wavfile => wavefile)

x_train = wavefile[0][:,:-1,:]                                   #x_train은 전체에서 마지막 column하나 뺀거
y_train = wavefile[0][:,1:,:]                                    #y_train은 맨앞에 하나 뺀거

for i in range (len(wavfile)-1):                                        #concatenate 해준다.
    x_train = np.concatenate((x_train,wavefile[i+1][:,:-1,:]), axis=0)
    y_train = np.concatenate((y_train,wavefile[i+1][:,1:,:]), axis=0)

#input = x_train[2]
#input = input[40000,:]
#input = input.reshape(1, 1, 256)
#z = np.zeros((1, timestep-1 , 256))
#input = np.concatenate((z,input), axis=1)
input = x_train[2]
#input = input[:-1,:]
input = input.reshape(1, timestep, 256)
updated_input = input


out = []
final_out = []
for i in range(timestep-25333):
    outputs = model.predict(updated_input, verbose=1)
    output = np.argmax(outputs[:,timestep-1,:])
    out.append(output)
    final_out.append(output)
    output = mu_to_onehot(out)
    output = output.reshape(1, 1, 256)
    updated_input = np.concatenate((updated_input, output), axis=1)
    updated_input = updated_input[:,1:,:]
    out.pop(0)
    print("Current step : ", i)
    print("Total ",(i*100/timestep)," Percent is succeeded")

print(final_out)
print(len(final_out))


################### INVERSE MU_LAW ################################################

mu = 256
inversed = []

def inv_mu_law(output):
    inversed.append((np.sign(output)*(1/mu)*(pow((1+mu),abs(output))-1))*pow(2,15))

for i in range(len(final_out)):
    final_out[i] = final_out[i]*2/255-1
    inv_mu_law(final_out[i])

f = open('generated_700.txt', 'w')
sp.dump(inversed, f)
f.close()

#print(inv_mu_law(output))


# plt.plot(inv_mu_law(output))
# plt.show()


