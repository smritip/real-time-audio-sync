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

#from livenote import LiveNote
from otw_eran import OnlineTimeWarping
from chroma import wav_to_chroma, wav_to_chroma_col

# from IMS:
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

class test_livenote_live(core.BaseWidget):

    def __init__(self):
        super(test_livenote_live, self).__init__()

        self.label = gfxutil.topleft_label()
        self.add_widget(self.label)

        ref = 'Songs/bso/bso_01.wav'
        ref_seq = wav_to_chroma(ref)

        params = {'c': 50, 'max_run_count': 3}
        debug_params = {'seq': False, 'all': False}

        self.ln = OnlineTimeWarping(ref_seq, params)

        self.ref_gt_times = []
        self.ref_gt_beats = []
        self.ref_gt_labels = []
        
        ref_song = ref[:-4]
        print "Reference song:", ref_song
        ref_csv = ref_song + '.csv'
     
        with open(ref_csv) as ref_csv_data:
            reader = csv.reader(ref_csv_data)
            for row in reader:
                self.ref_gt_times.append(float(row[0]))
                self.ref_gt_beats.append(int(row[1]))
                self.ref_gt_labels.append(str(row[2]))


        # getting audio from mic
        self.audio = audio.Audio(1, input_func=self.receive_audio)
        self.record = False
        self.input_buffers = []
        self.live_wave = None

        self.frame = 0
        self.fft_len = 4096
        self.hop_size = 2048

        self.beat = 0
        self.time = 0
        self.mus_label = ''

        self.start_time = 0
        self.stop = False

        # mic input levels
        self.mic_meter = MeterDisplay((50, 25),  150, (-96, 0), (.1,.9,.3))
        self.mic_graph = GraphDisplay((110, 25), 150, 300, (-96, 0), (.1,.9,.3))
        self.canvas.add(self.mic_meter)
        self.canvas.add(self.mic_graph)

        self.f = open("tests/livenote_test_live_" + str(time.time()) + ".txt", "w+")
        self.f.write("%s\r\n" % ref)
        self.f.write('fft_len: %d\r\n' % self.fft_len)
        self.f.write('hop_size: %d\r\n' % self.hop_size)
        self.f.write('search_band_width: %d\r\n' % params['c'])
        self.f.write('max_run_count: %d\r\n' % params['max_run_count'])

    def on_key_down(self, keycode, modifiers):
        # toggle recording
        if keycode[1] == 'r':
            self.record = not self.record
            if self.record:
                self.start_time = time.time()
            elif not self.record:
                self.sync_ests = self.ln.path
                for i in range(len(self.sync_ests)):
                    self.f.write("%d %d\r\n" % (self.sync_ests[i][0], self.sync_ests[i][1]))

    def on_update(self) :
        self.audio.on_update()
        self.label.text = "label:%s\r\n" % self.mus_label
        self.label.text += "beat:%.2f" % self.beat

    def receive_audio(self, frames, num_channels=1) :
        if self.record:
            self.input_buffers.append(frames)
            self.frame += len(frames)
            if self.frame >= self.fft_len:
                self.frame -= self.fft_len
                if not self.stop:
                    self._process_input()

        # Microphone volume level, take RMS, convert to dB.
        # display on meter and graph
        mono = frames
        rms = np.sqrt(np.mean(mono ** 2))
        rms = np.clip(rms, 1e-10, 1) # don't want log(0)
        db = 20 * np.log10(rms)      # convert from amplitude to decibels 
        self.mic_meter.set(db)
        self.mic_graph.add_point(db)

    def _process_input(self) :
        data = writer.combine_buffers(self.input_buffers)
        print 'live buffer size:', len(data), 'frames'
        self.live_wave = np.zeros((12, 1))
        first = True
        wave = np.zeros((12, 1))
        while len(data) >= self.fft_len and not self.stop:
            self.live_wave[:, 0] = wav_to_chroma_col(data[:self.fft_len])
            cont = self.ln.insert(self.live_wave[:, 0])
            if cont == "stop":
                print self.ln.path
                self.stop = True
            # else:
            #     wave[:, 0] = wav_to_chroma_col(data[:self.fft_len])
            #     print "SHAPES"
            #     print self.live_wave.shape
            #     print wave.shape
            #     self.live_wave = np.stack((self.live_wave, wave), axis=0)
            if len(self.ln.path) > 0:
                (beat, mus_label) = self.get_beat_and_label(self.ln.path[-1][1], self.ref_gt_times, self.ref_gt_beats, self.ref_gt_labels)
                if beat:
                    self.beat = beat
                if mus_label:
                    self.mus_label = mus_label
                delta = time.time() - self.start_time
                sample = delta / (2048 / 22050.)
                print "    ", sample
                print "    ", self.ln.path[-1][0], self.ln.path[-1][1]

            data = data[self.hop_size:]
        self.input_buffers = [data.tolist()]

    def get_beat_and_label(self, sample, gt_times, gt_beats, gt_labels):
        # convert sample to time
        time = sample * (2048 / 22050.)
        for i in range(len(gt_times)):
            if i == 0:
                if time <= gt_times[i]:
                    if gt_times[i] != 0:
                        frac = float(gt_times[i] - time) / (gt_times[i] - 0)
                    else:
                        frac = 0
                    return (gt_beats[i] - frac, gt_labels[0])
            else:
                if gt_times[i-1] <= time <= gt_times[i]:
                    frac = float(gt_times[i] - time) / (gt_times[i] - gt_times[i-1])
                    return (gt_beats[i] - frac, gt_labels[i-1])

        return (None, None)

core.run(test_livenote_live)