import numpy as np
import librosa
import csv
import os

from livenote import LiveNote
from livenote_v2 import LiveNoteV2
from wtw import WTW
from dtw import DTW
from chroma import wav_to_chroma, wav_to_chroma_col


def lines_from_file(filename):
    f = open(filename)
    return f.readlines()

def tokens_from_line(line):
    return line.strip().split('\t')

def data_from_file(filename):
    path = []
    lines = lines_from_file(filename)
    tokens =  [tokens_from_line(line) for line in lines[5:]]
    for t in tokens:  # this is the format of one line (time, label) in the gems/barlines text files
        l, r = t[0].split(" ")[0], t[0].split(" ")[1]
        path.append((int(l), int(r)))
    return path

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
        num_off1_secs = 0
        num_off3_secs = 0
        num_off5_secs = 0
        num_off10_secs = 0
        for (l, r) in self.path:
            l_beat = self.get_beat(l, self.live_gt_times, self.live_gt_beats)
            r_beat = self.get_beat(r, self.ref_gt_times, self.ref_gt_beats)
            if l_beat and r_beat:

                # get error in terms of beats
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

                # get error in terms of seconds
                seconds_off = self.get_secs_off(r_beat, l_beat)
                if seconds_off > 1:
                    num_off1_secs += 1
                if seconds_off > 3:
                    num_off3_secs += 1
                if seconds_off > 5:
                    num_off5_secs += 1
                if seconds_off > 10:
                    num_off10_secs += 1

                count += 1

        print "Percent incorrect (within 1 beat):", (float(num_off1) / count) * 100, "%"
        print "Percent incorrect (within 3 beats):", (float(num_off3) / count) * 100, "%"
        print "Percent incorrect (within 5 beats):", (float(num_off5) / count) * 100, "%"
        print "Percent incorrect (within 10 beats):", (float(num_off10) / count) * 100, "%"
        print "Percent incorrect (within 1 second):", (float(num_off1_secs) / count) * 100, "%"
        print "Percent incorrect (within 3 seconds):", (float(num_off3_secs) / count) * 100, "%"
        print "Percent incorrect (within 5 seconds):", (float(num_off5_secs) / count) * 100, "%"
        print "Percent incorrect (within 10 seconds):", (float(num_off10_secs) / count) * 100, "%"
        return (float(num_off3_secs) / count) * 100
                

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

    def get_time(self, beat):
        time = self.live_gt_times[int(beat)]
        if int(beat) + 1 < len(self.live_gt_times):
            time += (beat%1) * (self.live_gt_times[int(beat) + 1] - self.live_gt_times[int(beat)])
        return time

    def get_secs_off(self, ref_beat, live_beat):
        return abs(self.get_time(ref_beat) - self.get_time(live_beat))
        

params = {'search_band_width': 50, 'max_run_count': 3}
debug_params = {'seq': False, 'all': False}

class test_livenote():

    def __init__(self, ref, live, v2=False, path=None):
        ref_seq = wav_to_chroma(ref)
        live_seq = wav_to_chroma(live)

        if path is None:
            # print "initializing livenote"
            if v2:
                ln = LiveNoteV2(ref_seq, params, debug_params)
            else:
                ln = LiveNote(ref_seq, params, debug_params)

            for i in range(live_seq.shape[1]):
                cont = ln.insert(live_seq[:,i])
                if cont == "stop":
                    break
            ln_path = np.array(ln.path)
        else:
            ln_path = path

        # print "testing livenote"
        self.ln_test = test_simple(ref, live, ln_path)
    
    def evaluate(self):
        return self.ln_test.get_error()

wtw_params = {'fft_len': 4096, 'hop_size': 2048, 'dtw_win_size': 4096*10, 'dtw_hop_size': 2048*10}
wtw_debug_params = {'chroma': False, 'song': False, 'error': True, 'error_detail': False, 'alg': False}

class test_wtw():

    def __init__(self, ref, live):
        wtw = WTW(ref, wtw_params, wtw_debug_params)

        live_recording, fs = librosa.load(live)
        assert(fs == 22050)

        # print "calling insert"
        buffers = np.array_split(live_recording, 4096)
        for buf in buffers:
            cont = wtw.insert(buf.tolist())
            if cont == "stop":
                break
        wtw_path = np.array(wtw.path)

        # print "testing WTW"
        self.wtw_test = test_simple(ref, live, wtw_path)
    
    def evaluate(self):
        return self.wtw_test.get_error()

class test_all():  # multiple songs
    
    def __init__(self, recordings_dir, dtw_variant):
        self.recordings_dir = recordings_dir
        self.dtw_variant = dtw_variant
        
    def evaluate(self):
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
                for i in range(len(recs)):
                    for j in range(i, len(recs)):
                        if i != j:
                # for ref in recs:
                            ref = recs[i]
                            live = recs[j]
                            gt_ref = self.recordings_dir + d + '/' + ref + '.csv'
                            gt_live = self.recordings_dir + d + '/' + live + '.csv'
                            wav_ref = self.recordings_dir + d + '/' + ref + '.wav'
                            wav_live = self.recordings_dir + d + '/' + live + '.wav'
                            if self.dtw_variant == LiveNote:
                                test = test_livenote(wav_ref, wav_live)
                            elif self.dtw_variant == LiveNoteV2:
                                test = test_livenote(wav_ref, wav_live, v2=True)
                            elif self.dtw_variant == WTW:
                                test = test_wtw(wav_ref, wav_live)
                            elif self.dtw_variant == DTW:
                                test = test_dtw(wav_ref, wav_live)
                            error = test.evaluate()
                            errors.append(error)
                            # check against path from field testing too
                            if wav_ref == 'Songs/bso/bso_01.wav' and self.dtw_variant == LiveNote:
                                print "\n"
                                print "Field testing results (ie accuracy of path generated):"
                                live_path = data_from_file("tests/bso_livenote_test_live.txt")
                                live_test = test_livenote(wav_ref, wav_live, path=live_path)
                                live_error = live_test.evaluate()
                                errors.append(live_error)
                            print "\n"

            break  # break because only want dirs of root folder (the "Songs" folder)
            
        errors = np.array(errors)
        print "\n"
        print "Errors (percents incorrect within 3 seconds):\n", errors
        print "Average errors:\n", np.mean(errors)
        print "\n"
        print "\n"
        return np.mean(errors)

print "\n"
print "==================\n"
print " Testing Livenote\n"
print "==================\n\n"
test_ln = test_all('Songs/', LiveNote)
avg_error_ln = test_ln.evaluate()

print "\n"
print "=====================\n"
print " Testing Livenote V2\n"
print "=====================\n\n"
test_ln_v2 = test_all('Songs/', LiveNoteV2)
avg_error_ln_v2 = test_ln_v2.evaluate()

print "\n"
print "=============\n"
print " Testing WTW\n"
print "=============\n\n"
test_w = test_all('Songs/', WTW)
avg_error_wtw = test_w.evaluate()