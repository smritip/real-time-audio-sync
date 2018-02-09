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
from wtw import WTW

class test_single_recording():
    
    def __init__(self, dtw, ref_recording, live_recording, ref_ground_truth, live_ground_truth, params, debug_params):
        self.variant = dtw
        self.dtw = dtw(ref_recording, params, debug_params)  # passed in 'dtw', which is some form of DTW_x...
        
        self.live_recording, fs = librosa.load(live_recording)
        assert(fs == 22050)

        self.chroma_info = debug_params['chroma']
        self.song_info = debug_params['song']
        self.error_info = debug_params['error']
        self.error_detail = debug_params['error_detail']
        self.alg_info = debug_params['alg']
        
        if self.alg_info:
            print "Testing", self.variant
        
        self.ref_ground_truth_time = []
        self.ref_ground_truth_beats = []
        self.live_ground_truth_time = []
        self.live_ground_truth_beats = []
        
        ref_song = ref_recording[:-4]
        if self.song_info:
            print "Reference song:", ref_song
        ref_csv_file = ref_song + '.csv'
        live_song = live_recording[:-4]
        if self.song_info:
            print "Live song:", live_song
        live_csv_file = live_song + '.csv'
     
        with open(ref_csv_file) as ref_csv_data:
            reader = csv.reader(ref_csv_data)
            for row in reader:
                self.ref_ground_truth_time.append(float(row[0]))
                self.ref_ground_truth_beats.append(int(row[1]))
                
        with open(live_csv_file) as live_csv_data:
            reader = csv.reader(live_csv_data)
            for row in reader:
                self.live_ground_truth_time.append(float(row[0]))
                self.live_ground_truth_beats.append(int(row[1]))
        
    def evaluate(self, buf_size):
        '''Evaluate single piece of music with WTW.'''
        # Emulate live recording via creation of buffers
        buffers = np.array_split(self.live_recording, buf_size)
        # For each buffer, get the synchronization estimate (ie the estimated position)
        # via call to insert
        for buf in buffers:
            est = self.dtw.insert(buf.tolist())
            if est == "stop":
                break
        
        self.sync_ests = self.dtw.path
        
        if self.variant == LiveNote:
            if self.chroma_info:
                # show live chroma
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(self.dtw.chroma_live, y_axis='chroma', x_axis='time')
                plt.colorbar()
                plt.title('Live Chromagram')
                plt.tight_layout()
                # show distance matrix
                plt.figure()
                plt.imshow(self.dtw.D)
                plt.colorbar()
                plt.title('Distance Matrix')
                plt.tight_layout()

        # Compare estimates to ground truth, and return error
        error = self.get_error()
        return error
    
    def get_error(self):
        error = 0
        num_off1 = 0
        num_off3 = 0
        if self.error_detail:
            ff = float(self.dtw.fs) / self.dtw.hop_size
            gsamples = [x * ff for x in self.ref_ground_truth_time]
            print "samples at", gsamples
        for (l, r) in self.sync_ests:
            l_beat = self.get_beat(l, self.live_ground_truth_time, self.live_ground_truth_beats)
            r_beat = self.get_beat(r, self.ref_ground_truth_time, self.ref_ground_truth_beats)
            if self.error_detail:
                print "(l, r): ", l, r
                print "est: ", l * (self.dtw.hop_size / 22050.) , r * (self.dtw.hop_size / 22050.)
                print "beats:", l_beat, r_beat
            diff = (r_beat - l_beat)**2
            if abs(r_beat - l_beat) > 1:
                num_off1 += 1
            if abs(r_beat - l_beat) > 3:
                num_off3 += 1
            error += diff
        if self.error_info:
            print "Percent incorrect (within 1 beat):", (float(num_off1) / len(self.sync_ests)) * 100, "%"
        if self.error_detail:
            print "Percent incorrect (within 3 beats):", (float(num_off3) / len(self.sync_ests)) * 100, "%"
            print "Error:", error
        # TODO: fix re-def of error... right now it is percent incorrect within 1 beat
        error = (float(num_off1) / len(self.sync_ests)) * 100
        return error
    
    def get_beat(self, t, gtime, gbeats):
        ff = float(self.dtw.fs) / self.dtw.hop_size
        gsam = [x * ff for x in gtime]
        for i in range(len(gsam) - 1):
            if t < gsam[i]:
                return 0
            if gsam[i] <= t < gsam[i+1]:
                beatBefore = gbeats[i]
                timeBefore = gtime[i]
                timeAfter = gtime[i + 1]
                time = t / ff
                p_beat = (time - timeBefore) / (timeAfter - timeBefore)
                return beatBefore + p_beat
        return gbeats[-1]


class test_DTW():  # multiple songs, 1 DTW_x algorithm
    
    def __init__(self, dtw, recordings_dir, params, debug_params):
        self.recordings_dir = recordings_dir
        self.params = params
        self.debug_params = debug_params
        self.dtw = dtw
        
    def evaluate(self, buf_size):
        '''Evaluate a DTW variant (test with several pieces).'''
        errors = []
        # TODO: change following pseudocode to real code
        # each folder has multiple recordings of one song
        for subdir, dirs, files in os.walk(self.recordings_dir):
            for d in dirs:
                recs = []
                for subdir2, dirs2, files2 in os.walk('Songs/'+d):
                    for f in files2:
                        if f.startswith(d) and f[:-4] not in recs and not (f[:-4].endswith('_20b')):
                            recs.append(f[:-4])
                for ref in recs:
                    gt_ref = self.recordings_dir + d + '/' + ref + '.csv'
                    for live in recs:
                        gt_live = self.recordings_dir + d + '/' + live + '.csv'
                        wav_ref = self.recordings_dir + d + '/' + ref + '.wav'
                        wav_live = self.recordings_dir + d + '/' + live + '.wav'
                        test = test_single_recording(self.dtw, wav_ref, wav_live, gt_ref, gt_live, self.params, self.debug_params)
                        error = test.evaluate(buf_size)
                        errors.append(error)
                        if self.debug_params['error']:
                            print "\n"

            break  # break because only want dirs of root folder (the "Songs" folder)
            
        errors = np.array(errors)
        return np.mean(errors)


class test_all():  # multiple songs, multiple DTWs (test each DTW with multiple recordings)
    
    def __init__(self, dtws, recordings_dir, params, debug_params):
        self.recordings_dir = recordings_dir
        self.params = params
        self.debug_params = debug_params
        self.dtw = dtws
        
    def evaluate(self):
        '''Evaluate all DTW variants (with all pieces).'''
        errors = {}
        for dtw in self.dtws:
            test = test_DTW(dtw, self.recordings_dir, self.params, self.debug_params)
            error = test.evaluate()
            errors[dtw] = error
            
        return errors