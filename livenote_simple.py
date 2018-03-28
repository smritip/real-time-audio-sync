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
import time

from chroma import *

class LiveNote():
    
    def __init__(self, ref, params, debug_params):
        
        # algorithm params
        self.search_band_width = params['search_band_width']  # max lookback
        self.max_run_count = params['max_run_count']  # max slope
        
        self.seq_info = debug_params['seq']
        self.debug = debug_params['all']
        
        self.seq_ref = ref

        if self.seq_info or self.debug:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(self.seq_ref, y_axis='chroma', x_axis='time')
            plt.colorbar()
            plt.title('Reference Chromagram')
            plt.tight_layout()
        
        # initialize arrays and matrices for live audio
        # double length of ref seq for live seq to make sure there is enough space w/out dynamically changing
        self.N = self.seq_ref.shape[1] * 2  # rows are live
        self.M = self.seq_ref.shape[1]      # cols are ref
        #self.cost = np.zeros((self.N, self.M))
        self.cost = -1 * np.ones((self.N, self.M))
        self.acc_cost = np.empty((self.N, self.M))
        self.acc_cost.fill(float('inf'))
        
        self.F = self.seq_ref.shape[0]  # number of features (should be 12 for chroma)
        self.seq_live = np.zeros((self.F, self.N))
        
        # TODO: use pyaudio buffer
        self.buf = np.zeros((self.F, self.N))
        self.path = []
        
        # pointers & variables for LiveNote algorithm
        self.ref_ptr = 0
        self.live_ptr = 0
        self.previous = None
        self.run_count = 0
        self.continueFlag = False
        self.prev_inc = None
        self.input_ptr = 0

        self.wav_to_seq_live = wav_to_chroma_col
        
        if self.seq_info:
            self.seq_ptr = 0

    def fill_input(self, live):
        self.seq_live[:,self.live_ptr] = live[:,self.live_ptr]

    # def insert_input(self, buf):
    #     #self.buf += buf
    #     n = buf.shape[1]
    #     self.seq_live[:, self.input_ptr:self.input_ptr + n] = buf
    #     self.input_ptr += n
    #     # if len(self.buf) >= self.min_len:
    #     #     self.run_alg()

    # def check_input(self):
    #     if len(self.buf) >= 4096:
    #         return True
    #     return False

    # def get_input(self):
    #     # self.seq_live[:,self.live_ptr] = self.wav_to_seq_live(self.buf[:4096])
    #     # self.buf = self.buf[4096:]
    #     print self.live_ptr
    #     self.seq_live[:, self.live_ptr] = self.buf[:, self.live_ptr]

    # def run_alg(self):
    #     while not self.check_input:
    #         time.sleep(0.5)

    #     self.get_input()

    #     self.eval_path_cost(self.live_ptr, self.ref_ptr)

    #     while True:

    #         inc = self.get_inc()

    #         if self.debug:
    #             print "inc:", inc
    #             print "ref ptr:", self.ref_ptr
    #             print "live ptr:", self.live_ptr
                
    #         # process a row
    #         if inc != "column":
    #             self.live_ptr += 1
    #             while not self.check_input:
    #                 time.sleep(0.5)
    #             self.get_input()
    #             k1 = max(0, self.ref_ptr - self.search_band_width + 1)
    #             k2 = self.ref_ptr + 1
    #             for k in range(k1, k2):
    #                 if self.debug:
    #                    print "live, k:", self.live_ptr, k
    #                 self.eval_path_cost(self.live_ptr, k)
                    

    #         # process a column
    #         if inc != "row":
    #             self.ref_ptr += 1
    #             k1 = max(0, self.live_ptr - self.search_band_width + 1)
    #             k2 = self.live_ptr + 1
    #             for k in range(k1, k2):
    #                 if self.debug:
    #                    print "k, ref:", k, self.ref_ptr
    #                 self.eval_path_cost(k, self.ref_ptr)

    #         if inc == self.previous:
    #             self.run_count += 1
    #             if self.debug:
    #                 print "increasing run count to", self.run_count
    #         else:
    #             self.run_count = 1
    #             if self.debug:
    #                 print "run count is now", self.run_count

    #         if inc != "both":
    #             self.previous = inc


    def set_live(self, live):
        self.fill_input(live)

        # update cost matrix and accumulated cost matrix
        self.eval_path_cost(self.live_ptr, self.ref_ptr)

        while True:
        #while self.ref_ptr < self.M and self.live_ptr < self.N: 
            
            # decide
            # continue processing row (after getting more input, see below)
            inc = self.get_inc()

            if self.debug:
                print "inc:", inc
                print "ref ptr:", self.ref_ptr
                print "live ptr:", self.live_ptr
                
            # process a row
            if inc != "column":
                self.live_ptr += 1
                self.fill_input(live)
                k1 = max(0, self.ref_ptr - self.search_band_width + 1)
                k2 = self.ref_ptr + 1
                for k in range(k1, k2):
                    if self.debug:
                       print "live, k:", self.live_ptr, k
                    self.eval_path_cost(self.live_ptr, k)
                    

            # process a column
            if inc != "row":
                self.ref_ptr += 1
                k1 = max(0, self.live_ptr - self.search_band_width + 1)
                k2 = self.live_ptr + 1
                for k in range(k1, k2):
                    if self.debug:
                       print "k, ref:", k, self.ref_ptr
                    self.eval_path_cost(k, self.ref_ptr)

            if inc == self.previous:
                self.run_count += 1
                if self.debug:
                    print "increasing run count to", self.run_count
            else:
                self.run_count = 1
                if self.debug:
                    print "run count is now", self.run_count

            if inc != "both":
                self.previous = inc


    # def insert(self, live_audio_buf):
    #     # store incoming music
    #     self.buf += live_audio_buf
        
    #     # LiveNote algorithm:
    #     # identify window for one seq col ('INPUT u(t)' in pseudocode)
    #     while len(self.buf) >= self.min_len:

    #         seq = self.wav_to_seq_live(self.buf[:self.min_len])

    #         self.buf = self.buf[self.min_len:]
            
    #         if self.seq_info:
    #             self.seq_live[:, self.seq_ptr] = seq
    #             self.seq_ptr += 1
    
    #         # update cost matrix and accumulated cost matrix
    #         self.eval_path_cost(self.live_ptr, self.ref_ptr)
            
    #         # main loop of algorithm -- fill out matrices and search as needed
    #         while True:
    #         #while self.ref_ptr < self.M and self.live_ptr < self.N: 
                
    #             # decide
    #             # continue processing row (after getting more input, see below)
    #             if self.continueFlag:
    #                 inc = self.prev_inc
    #                 k1 = max(0, self.ref_ptr - self.search_band_width + 1)
    #                 k2 = self.ref_ptr + 1
    #                 for k in range(k1, k2):
    #                     if self.debug:
    #                        print "live, k:", self.live_ptr, k
    #                     self.eval_path_cost(self.live_ptr, k)
    #             else:
    #                 inc = self.get_inc()

    #             if self.debug:
    #                 print "inc:", inc
    #                 print "ref ptr:", self.ref_ptr
    #                 print "live ptr:", self.live_ptr
                    
    #             # process a row
    #             if inc != "column":
    #                 if not self.continueFlag:
    #                     self.live_ptr += 1
    #                     self.continueFlag = True
    #                     self.prev_inc = inc
    #                     break  # ('INPUT u(t)' in pseudocode)
    #                 if self.continueFlag:
    #                     self.continueFlag = False

    #             # process a column
    #             if inc != "row":
    #                 self.ref_ptr += 1
    #                 k1 = max(0, self.live_ptr - self.search_band_width + 1)
    #                 k2 = self.live_ptr + 1
    #                 for k in range(k1, k2):
    #                     if self.debug:
    #                        print "k, ref:", k, self.ref_ptr
    #                     self.eval_path_cost(k, self.ref_ptr)

    #             if inc == self.previous:
    #                 self.run_count += 1
    #                 if self.debug:
    #                     print "increasing run count to", self.run_count
    #             else:
    #                 self.run_count = 1
    #                 if self.debug:
    #                     print "run count is now", self.run_count

    #             if inc != "both":
    #                 self.previous = inc

                
    ########################
    ##  Helper functions  ##
    ########################
    
    def eval_path_cost(self, i, j):
        # update cost matrix
        self.cost[i, j] = 1 - np.dot(self.seq_live[:,i], self.seq_ref[:,j])

        # update accumulated cost matrix
        # initial condition
        if i == 0 and j == 0:
            self.acc_cost[i, j] = self.cost[i, j]
            return

        # regular case
        costs = []
        if i > 0:
            costs.append(self.acc_cost[i-1, j] + self.cost[i, j])
        if j > 0:
            costs.append(self.acc_cost[i, j-1] + self.cost[i, j])
        if i > 0 and j > 0:
            costs.append(self.acc_cost[i-1, j-1] + 2 * self.cost[i, j])
        
        if costs != []:
            best_cost = min(costs)
            self.acc_cost[i, j] = best_cost

        # else, self.acc_cost[i, j] will remain 'inf'    
        

    def get_inc(self):
        # first get best point (moved from end of pseudocode)
        (x, y) = self.calc_best_point()

        # update path with best point
        self.path.append((x, y))

        if self.live_ptr < self.search_band_width:
            if self.debug:
                print "both because live ptr less than search band"
            return "both"
        
        if self.run_count >= self.max_run_count:
            if self.previous == "row":
                return "column"
            else:
                return "row"
        
        if self.debug:
            print "x, y calculated:", x, y
            print "with ref ptr and live ptr:", self.ref_ptr, self.live_ptr

        if x < self.live_ptr:
            return "column"

        elif y < self.ref_ptr:
            return "row"

        if self.debug:
            print "both because last option"
        return "both"

    def calc_best_point(self):
        # from paper pseudocode:
        # (y,x) := argmin(pathCost(k,l)), where (k == t) or (l == j)

        ref1 = max(0, self.ref_ptr - self.search_band_width + 1)
        ref2 = self.ref_ptr + 1
        best_ref = ref1 + np.argmin(self.acc_cost[self.live_ptr, ref1:ref2])
        cost_ref = self.acc_cost[self.live_ptr, best_ref]

        live1 = max(0, self.live_ptr - self.search_band_width + 1)
        live2 = self.live_ptr + 1
        best_live = live1 + np.argmin(self.acc_cost[live1:live2, self.ref_ptr])
        cost_live = self.acc_cost[best_live, self.ref_ptr]

        if cost_ref < cost_live:
            return (self.live_ptr, best_ref)
        else:
            return (best_live, self.ref_ptr)
    

# params = {'search_band_width': 10, 'max_run_count': 3}
# debug_params = {'seq': False, 'all': False}
# ref = 'Songs/chopin/chopin_rubinstein_20b.wav'
# ln = LiveNote(ref, params, debug_params)

# live_music, fs = librosa.load('Songs/chopin/chopin_rachmaninoff_20b.wav')

# for i in range(0, 100):
#     buf = live_music[4096*i:(4096*(i+1))].tolist()
#     ln.insert(buf)