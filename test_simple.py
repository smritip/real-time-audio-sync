import numpy as np
import math
import os

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

from livenote import LiveNote
from otw_eran import OnlineTimeWarping
from wtw import WTW
from dtw import DTW
from chroma import *

class test_simple():

    def __init__(self, ref, live, path):

        self.ref_gt_times = []
        self.ref_gt_beats = []
        self.live_gt_times = []
        self.live_gt_beats = []

        self.path = path
        
        ref_song = ref[:-4]
        print "Reference song:", ref_song
        ref_csv = ref_song + '.csv'
        live_song = live[:-4]
        print "Live song:", live_song
        live_csv = live_song + '.csv'
     
        with open(ref_csv) as ref_csv_data:
            reader = csv.reader(ref_csv_data)
            for row in reader:
                self.ref_gt_times.append(float(row[0]))
                self.ref_gt_beats.append(int(row[1]))
                
        with open(live_csv) as live_csv_data:
            reader = csv.reader(live_csv_data)
            for row in reader:
                self.live_gt_times.append(float(row[0]))
                self.live_gt_beats.append(int(row[1]))

    def get_error(self):
        error = 0
        num_off1 = 0
        num_off3 = 0
        num_off5 = 0
        num_off10 = 0
        count = 0
        for (l, r) in self.path:
            l_beat = self.get_beat(l, self.live_gt_times, self.live_gt_beats)
            r_beat = self.get_beat(r, self.ref_gt_times, self.ref_gt_beats)
            if l_beat and r_beat:
                diff = abs(l_beat - r_beat)
                error += diff ** 2

                if diff > 1:
                    num_off1 += 1
                if diff > 3:
                    num_off3 += 1
                if diff > 5:
                    num_off5 += 1
                if diff > 10:
                    num_off10 += 1

                count += 1

                #print l_beat, r_beat, diff

        print "Percent incorrect (within 1 beat):", (float(num_off1) / count) * 100, "%"
        print "Percent incorrect (within 3 beat):", (float(num_off3) / count) * 100, "%"
        print "Percent incorrect (within 5 beat):", (float(num_off5) / count) * 100, "%"
        print "Percent incorrect (within 10 beat):", (float(num_off10) / count) * 100, "%"
        return error
                

    def get_beat(self, sample, gt_times, gt_beats):
        # convert sample to time
        time = sample * (2048 / 22050.)
        for i in range(len(gt_times)):
            if i == 0:
                if time <= gt_times[i]:
                    if gt_times[i] != 0:
                        frac = float(gt_times[i] - time) / (gt_times[i] - 0)
                    else:
                        frac = 0
                    return gt_beats[i] - frac
            else:
                if gt_times[i-1] <= time <= gt_times[i]:
                    frac = float(gt_times[i] - time) / (gt_times[i] - gt_times[i-1])
                    return gt_beats[i] - frac

        return None


ref = 'Songs/chopin/chopin_rubinstein.wav'
live = 'Songs/chopin/chopin_rachmaninoff.wav'
# ref = 'Songs/bach/bach_01.wav'
# live = 'Songs/bach/bach_02.wav'
ref_seq = wav_to_chroma(ref)
live_seq = wav_to_chroma(live)

# Smriti with set_live
print "initializing livenote"
params = {'search_band_width': 10, 'max_run_count': 3}
debug_params = {'seq': False, 'all': False}
ln = LiveNote(ref_seq, params, debug_params)

print "calling set_live"
ln.set_live(live_seq)
ln_path = np.array(ln.path)

print "testing livenote"
ln_test = test_simple(ref, live, ln_path)
ln_test.get_error()

print '\n'

# Smriti with insert
print "initializing livenote"
ln2 = LiveNote(ref_seq, params, debug_params)

print "calling insert"
for i in range(live_seq.shape[1]):
    cont = ln2.insert(live_seq[:,i])
    if cont == "stop":
        break
ln2_path = ln2.path

print "testing livenote"
ln2_test = test_simple(ref, live, ln2_path)
ln2_test.get_error()

print '\n'

# Eran with set_live
print "initializing OTW"
otw_params = {'c': 10, 'max_run_count': 3}
otw = OnlineTimeWarping(ref_seq, otw_params)

print "calling set_live"
otw.set_live(live_seq)
otw_path = otw.path

print "testing OTW"
otw_test = test_simple(ref, live, otw_path)
otw_test.get_error()

print '\n'

# Eran with insert
print "initializing OTW"
otw2_params = {'c': 10, 'max_run_count': 3}
otw2 = OnlineTimeWarping(ref_seq, otw2_params)

print "calling insert"
for i in range(live_seq.shape[1]):
    cont = otw2.insert(live_seq[:,i])
    if cont == "stop":
        break
otw2_path = otw2.path

print "testing OTW"
otw2_test = test_simple(ref, live, otw2_path)
otw2_test.get_error()

print '\n'

# WTW
print "initializing WTW"
wtw_params = {'fft_len': 4096, 'hop_size': 2048, 'dtw_win_size': 4096*10, 'dtw_hop_size': 2048*10}
wtw_debug_params = {'chroma': False, 'song': False, 'error': True, 'error_detail': False, 'alg': False}
wtw = WTW(ref, wtw_params, wtw_debug_params)

print "setting live music"
live_recording, fs = librosa.load(live)
assert(fs == 22050)

print "calling insert"
buffers = np.array_split(live_recording, 4096)
for buf in buffers:
    cont = wtw.insert(buf.tolist())
    if cont == "stop":
        break
wtw_path = np.array(wtw.path)

print "testing WTW"
wtw_test = test_simple(ref, live, wtw_path)
wtw_test.get_error()

print '\n'

# DTW
print "running DTW"
dtw_cost, dtw_acc_cost, dtw_path = DTW(live_seq, ref_seq)
print "testing DTW"
dtw_test = test_simple(ref, live, dtw_path)
dtw_test.get_error()