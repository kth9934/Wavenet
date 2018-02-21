from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import simplejson as sp
import wave
import librosa
from taehyoung_util import *
from wavenet_TH_models import path, wavfile
#from sy_model import *

################ MODEL LOAD & INPUT SET #################################

model = load_model('WaveNet_TH.h5')
model.load_weights('Weights_TH.h5')
#model = load_model('WaveNet_TH_drop.h5', custom_objects={'GatedActivation': GatedActivation}) <- custom layer 불러올때


input = load_to_onehot("sample5.wav",'',256) # 2-D
input_sample = np.expand_dims(input,0) # 3-D [1, input.shape]
#input_sample = np.expand_dims(input,0) # 3-D [1, input.shape]
input_sample = input_sample[:,5000:15000,:]
updated_input = input_sample

#################### GENERATION #############################################
final_out = []
for i in range(16000):
    outputs = model.predict(updated_input, verbose=1)
    output = np.argmax(outputs[0][-1])
    # out.append(output)
    final_out.append(output)
    out_onehot = np.zeros([1,1,256])
    out_onehot[0][0][output] = 1
    # output = output.reshape(1, 1, 256)
    updated_input = np.concatenate((updated_input, out_onehot), axis=1)
    updated_input = np.expand_dims(updated_input[0][1:],0)
    # out.pop(0)
    print(output)
    print("Current step : ", i)
    print("Total ",(i*100/16000)," Percent is succeeded")

#print(final_out)
#print(len(final_out))


################### INVERSE MU_LAW ################################################

mu = 256
inversed = []

def inv_mu_law(output):
    inversed.append((np.sign(output)*(1/mu)*(pow((1+mu),abs(output))-1))*pow(2,15))

    for i in range(len(final_out)):
        final_out[i] = final_out[i]*2/255-1
        inv_mu_law(final_out[i])

inversed = np.array(inversed)
librosa.output.write_wav('result_final', inversed, 16000, norm=False)

#print(inv_mu_law(output))


# plt.plot(inv_mu_law(output))
# plt.show()l


