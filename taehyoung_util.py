import numpy as np
import wave
import os
from glob import glob
from os import listdir
from os.path import isfile, join
import struct
import matplotlib.pyplot as plt
from librosa.core import load, resample
from scipy.io import wavfile

def load_audio(wav, path):
    # wav = wavfile
    raw, fs = load(os.path.join(path,wav), sr=16000, mono=True)         #fs = sample rate, raw는 값
    #frames = resample(raw, fs, 10000)

    return raw



def mu_quantizing(raw):
    quantized = []
    for f in raw:
        mu_padded = np.sign(f)*np.log(1+255*np.absolute(f))/np.log(256)  # mu-law 함수에 값을 집어넣음
        if (mu_padded > 1 ):
            mu_padded = 1
        elif (mu_padded < -1):
            mu_padded = -1
        quantized.append(int((mu_padded+1)*255/2)) # 집어넣은 후 quantizing해주기 위해 0~255범위로 표현
    return quantized

#frames = load_audio("sample.wav", "")
# plt.plot(frames) #기존의 input 확인
# plt.show()

#plt.plot(mu_quantizing(frames))
#plt.show()         # 퀀타이징 확인

def mu_to_onehot(quantized):
    onehot = np.zeros([len(quantized), 256], dtype='int32')   # 행은 quantized 된 input의 길이만큼, 열은 0~255까지 256개.
    for i, m in enumerate(quantized):          # for i, m in enumerate(quantized)에서 print(i,m)을 하면,
                                               # (i, quantized[i]) 가 출력된다.
        onehot[i][m] = 1
    return onehot

# quantized = mu_quantizing(frames) #onehot 확인
# print(mu_to_onehot(quantized))

def load_to_onehot(wavfile, path , dim): #concatenate several stages
     frames = load_audio(wavfile, path)
     quantized = mu_quantizing(frames)
     onehot = mu_to_onehot(quantized)
     return onehot


# def raw_audio(path, vals):                                       #trainset 가져오는 함수
#
#     files = [f for f in listdir(path) if isfile(join(path, f))]
#
#     for i in range(len(files)):
#         samples = load_to_onehot(files[i] ,path, 256)
#         vals.append(samples)
#
#     return vals




# print(load_to_onehot("sample.wav", "", 256))    최종확인

#print(load_to_onehot("sample.wav", "", 256).shape)
