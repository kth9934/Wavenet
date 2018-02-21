import wave
import os
from glob import glob
from os import listdir
from os.path import isfile, join
import struct
import matplotlib.pyplot as plt
from librosa.core import load, resample
from scipy.io import wavfile
import numpy as np

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


def load_generator(all_files):
    for file in all_files:
        file_name = file.split('/')
        file_name = file_name[-1].replace('.wav', '')
        raw = load_audio(file, '')
        mu_q = mu_quantizing(raw)
        mu_q = np.asarray(mu_q)

        yield file_name, mu_q

def save_wav_to_arr(data_dir):
    # VCTK -> 11000 files
    all_files = glob(os.path.join('../vctk/VCTK-Corpus/trainset_wav', '*wav'))
    output_dir = './trainset_np/'
    i = 1
    for file_name, audio_vector in load_generator(all_files):
        np.save(output_dir + file_name, audio_vector)
        print('save file : ' + file_name)
        print('Total ',i,'/ 11000 arrays are saved.')
        i = i+1

    print('save_wav_to_arr done.')
