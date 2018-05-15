import numpy as np
import math

# matplotlib for displaying the output
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)

# and IPython.display for audio output
import IPython.display as ipd

# Librosa for audio
import librosa
# And the display module for visualization
import librosa.display

import pyaudio
import csv

# globals
fft_len = 4096
hop_size = 2048
fs = 22050

# TODO: check additions (tuning, normalization, etc.)
def wav_to_chroma(path_to_wav):
    # generate wav
    wav, wav_fs = librosa.load(path_to_wav)
    assert(wav_fs == 22050)

    # create chroma (STFT --> spectrogram --> chromagram)
    stft = create_stft(wav)

    return create_chroma(stft)

def wav_to_chroma_col(wav_buf):
    assert(len(wav_buf) == fft_len)

    section = np.array(wav_buf)
    win = section * np.hanning(len(section))
    dft = np.fft.rfft(win)

    return create_chroma(dft)

def create_stft(wav):
    L = fft_len
    H = hop_size
    
    # use centered window by zero-padding
    x = np.concatenate((np.zeros(L/2), wav))
    
    N = len(x)
    
    num_bins = 1 + L/2
    num_hops = int(((N - L)/H) + 1)
    
    stft = np.empty((num_bins, num_hops), dtype=complex)
    
    M = num_hops

    for m in range(M):
        section = x[(m*H):((m*H) + L)]
        win = section * np.hanning(len(section))
        stft[:, m]= np.fft.rfft(win)
    
    return stft

def create_chroma(ft, normalize=True):
    spec = np.abs(ft)**2
    chromafb = librosa.filters.chroma(fs, fft_len)
    raw_chroma = np.dot(chromafb, spec)
    if not normalize:
        return raw_chroma
    else:
        chroma = librosa.util.normalize(raw_chroma, norm=2, axis=0)
        return chroma

def wav_to_chroma_diff(path_to_wav):
    # generate wav
    wav, wav_fs = librosa.load(path_to_wav)
    assert(wav_fs == 22050)

    # create chroma (STFT --> spectrogram --> chromagram)
    stft = create_stft(wav)

    chroma = create_chroma(stft, normalize=True)
    # print chroma[:20]
    # print np.diff(chroma)[:20]

    chroma_diff = np.diff(chroma)
    return np.clip(chroma_diff, 0, float('inf'))