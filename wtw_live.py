import numpy as np
import math

import librosa

import pyaudio
import csv
import time

from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Rectangle, Line
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate

from ims import audio, core, writer, gfxutil
from wtw import WTW

# from IMS:
class WaveArray(object):
    def __init__(self, np_array, num_channels):
        super(WaveArray, self).__init__()

        self.data = np_array
        self.num_channels = num_channels

    # start and end args are in units of frames,
    # so take into account num_channels when accessing sample data
    def get_frames(self, start_frame, end_frame) :
        start_sample = start_frame * self.num_channels
        end_sample = end_frame * self.num_channels
        return self.data[start_sample : end_sample]

    def get_num_channels(self):
        return self.num_channels

# graphical display of a meter
class MeterDisplay(InstructionGroup):
    def __init__(self, pos, height, in_range, color):
        super(MeterDisplay, self).__init__()
        
        self.max_height = height
        self.range = in_range

        # dynamic rectangle for level display
        self.rect = Rectangle(pos=(1,1), size=(50,self.max_height))

        self.add(PushMatrix())
        self.add(Translate(*pos))

        # border
        w = 52
        h = self.max_height+2
        self.add(Color(1,1,1))
        self.add(Line(points=(0,0, 0,h, w,h, w,0, 0,0), width=2))

        # meter
        self.add(Color(*color))
        self.add(self.rect)

        self.add(PopMatrix())

    def set(self, level):
        h = np.interp(level, self.range, (0, self.max_height))
        self.rect.size = (50, h)

# continuous plotting and scrolling line
class GraphDisplay(InstructionGroup):
    def __init__(self, pos, height, num_pts, in_range, color):
        super(GraphDisplay, self).__init__()

        self.num_pts = num_pts
        self.range = in_range
        self.height = height
        self.points = np.zeros(num_pts*2, dtype = np.int)
        self.points[::2] = np.arange(num_pts) * 2
        self.idx = 0
        self.mode = 'scroll'
        self.line = Line( width = 1 )
        self.add(PushMatrix())
        self.add(Translate(*pos))
        self.add(Color(*color))
        self.add(self.line)
        self.add(PopMatrix())

    def add_point(self, y):
        y = int( np.interp( y, self.range, (0, self.height) ))

        if self.mode == 'loop':
            self.points[self.idx + 1] = y
            self.idx = (self.idx + 2) % len(self.points)

        elif self.mode == 'scroll':
            self.points[3:self.num_pts*2:2] = self.points[1:self.num_pts*2-2:2]
            self.points[1] = y

        self.line.points = self.points.tolist()


class test_single_recording_WTW_live(core.BaseWidget):
    
    def __init__(self):
        super(test_single_recording_WTW_live, self).__init__()

        self.label = gfxutil.topleft_label()
        self.add_widget(self.label)

        params = {'fft_len': 4096, 'hop_size': 2048, 'dtw_win_size': 4096*50, 'dtw_hop_size': 2048*50}
        debug_params = {'chroma': True, 'song': True, 'error': True, 'error_detail': False, 'alg': True}
        self.ref_recording = 'Songs/chopin/chopin_rubinstein_20b.wav'
        self.live_recording = 'Songs/chopin/chopin_rachmaninoff_20b.wav'
        #self.live_recording = None


        self.dtw = WTW(self.ref_recording, params, debug_params)
        # self.live_recording, fs = librosa.load(live_recording)
        # assert(fs == 22050)

        self.song_info = debug_params['song']
        self.error_info = debug_params['error']
        self.error_detail = debug_params['error_detail']
        
        self.ref_gt_times = []
        self.ref_gt_beats = []
        self.live_gt_times = []
        self.live_gt_beats = []
        
        ref_song = self.ref_recording[:-4]
        print "Reference song:", ref_song
        ref_csv = ref_song + '.csv'
     
        with open(ref_csv) as ref_csv_data:
            reader = csv.reader(ref_csv_data)
            for row in reader:
                self.ref_gt_times.append(float(row[0]))
                self.ref_gt_beats.append(int(row[1]))
                
        if self.live_recording:
            live_song = self.live_recording[:-4]
            print "Live song:", live_song
            live_csv = live_song + '.csv'
            with open(live_csv) as live_csv_data:
                reader = csv.reader(live_csv_data)
                for row in reader:
                    self.live_gt_times.append(float(row[0]))
                    self.live_gt_beats.append(int(row[1]))
        
    	# for live testing, getting audio from mic:
    	self.audio = audio.Audio(1, input_func=self.receive_audio)
    	self.record = False
        self.input_buffers = []
        self.live_wave = None

        self.frame = 0
        self.fft_len = params['fft_len']

        self.path_size = 0
        self.ref_beat = 0
        self.live_beat = 0
        self.live_time = 0

        self.start_time = 0
        self.end_time = 0

        # mic input levels
        self.mic_meter = MeterDisplay((50, 25),  150, (-96, 0), (.1,.9,.3))
        self.mic_graph = GraphDisplay((110, 25), 150, 300, (-96, 0), (.1,.9,.3))
        self.canvas.add(self.mic_meter)
        self.canvas.add(self.mic_graph)

        self.f = open("tests/wtw_test_live_" + str(time.time()) + ".txt", "w+")
        self.f.write("%s\r\n" % self.ref_recording)
        self.f.write('fft_len: %d\r\n' % params['fft_len'])
        self.f.write('hop_size: %d\r\n' % params['hop_size'])
        self.f.write('dtw_win_size: %d\r\n' % params['dtw_win_size'])
        self.f.write('dtw_hop_size: %d\r\n' % params['dtw_hop_size'])

    def on_update(self) :
        self.audio.on_update()
        self.label.text = "ref beat:%.2f" % self.ref_beat
        self.label.text += "\nlive beat:%.2f" % self.live_beat
        self.label.text += "\nlive time:%.2f" % self.live_time

    def receive_audio(self, frames, num_channels=1) :
        if self.record:
            self.input_buffers.append(frames)
            self.frame += len(frames)
            # TODO do not keep track of frame here
            #if self.frame >= self.fft_len or not self.record:
            if self.frame >= self.fft_len:
            	self.frame -= self.fft_len
            	self._process_input()

        # Microphone volume level, take RMS, convert to dB.
        # display on meter and graph
        mono = frames
        rms = np.sqrt(np.mean(mono ** 2))
        rms = np.clip(rms, 1e-10, 1) # don't want log(0)
        db = 20 * np.log10(rms)      # convert from amplitude to decibels 
        self.mic_meter.set(db)
        self.mic_graph.add_point(db)

    def on_key_down(self, keycode, modifiers):
        # toggle recording
        if keycode[1] == 'r':
            self.record = not self.record
            if self.record:
                self.start_time = time.time()
            elif not self.record:
                self.sync_ests = self.dtw.path
                for i in range(len(self.sync_ests)):
                    self.f.write("%d %d\r\n" % (self.sync_ests[i][0], self.sync_ests[i][1]))

        if keycode[1] == 'e':
            if self.live_recording:
                self.error = self.get_error()

    def _process_input(self) :
        data = writer.combine_buffers(self.input_buffers)
        print 'live buffer size:', len(data), 'frames'
        #write_wave_file(data, NUM_CHANNELS, 'recording.wav')
        #np.save("rec.npy", data)
        #self.live_wave = WaveArray(data, 1)
        self.live_wave = data
        self.input_buffers = []
        self.evaluate()

    def evaluate(self):
        '''Evaluate single piece of music with WTW.'''
        print "inserting"
        self.dtw.insert(self.live_wave.tolist())
        # TODO: get best estimate of corresponding point in reference recording (units of time)
        # TODO: seconds + floating point beats of reference audio
        # TODO: play and pause
        # TODO: check out audio source
        # path_size = len(self.dtw.path)
        # if path_size > self.path_size:
            #for i in range(self.path_size)
        # print self.dtw.path
        # if len(self.dtw.path) > 0:
        #     self.ref_beat = self.get_beat(self.dtw.path[-1][1], self.ref_gt_times, self.ref_gt_beats)
        #     self.live_beat = self.get_beat(self.dtw.path[-1][0], self.live_gt_times, self.live_gt_beats, live=True)
        # self.path_size = path_size
        if len(self.dtw.path) > 0:
            self.live_beat = self.get_beat(self.dtw.path[-1][0], self.live_gt_times, self.live_gt_beats, live=True)
            delta = time.time() - self.start_time
            sample = delta / (2048 / 22050.)
            print "    ", sample
            print "    ", self.dtw.path[-1][0], self.dtw.path[-1][1]

    def get_beat_from_time(self, time, gt_times, gt_beats, live=False):
        if live:
            self.live_time = time
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
    
    def get_error(self):
        error = 0
        num_off1 = 0
        num_off3 = 0
        num_off5 = 0
        num_off10 = 0
        count = 0
        for (l, r) in self.sync_ests:
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

        string_1 = "Percent incorrect (within 1 beat):" + str((float(num_off1) / count) * 100) + "%\r\n"
        string_3 = "Percent incorrect (within 3 beats):" + str((float(num_off3) / count) * 100) + "%\r\n"
        string_5 = "Percent incorrect (within 5 beats):" + str((float(num_off5) / count) * 100) + "%\r\n"
        string_10 = "Percent incorrect (within 10 beats):" + str((float(num_off10) / count) * 100) + "%\r\n"

        self.f.write(string_1)
        self.f.write(string_3)
        self.f.write(string_5)
        self.f.write(string_10)

        return error
    
    def get_beat(self, sample, gt_times, gt_beats, live=False):
        # convert sample to time
        time = sample * (2048 / 22050.)
        if live:
            self.live_time = time
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


core.run(test_single_recording_WTW_live)