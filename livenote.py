class LiveNote():
    
    def __init__(self, ref_recording, params, debug_params):
        # reference audio, fs = 22050
        self.ref, self.fs = librosa.load(ref_recording)
        assert(self.fs == 22050)
        
        # params
        self.fft_len = params['fft_len']
        self.hop_size = params['hop_size']
        self.search_band_width = params['search_band_width']
        self.max_run_count = params['max_run_count']  # max slope
        
        self.chroma_info = debug_params['chroma']
        
        # create STFT, spectrogram, and chromagram of reference audio
        # TODO: check additions (tuning, normalization, etc.)
        # note: using own stft function (vs. librosa's) b/c insert will also use self.stft
        stft_ref = self.stft(self.ref, self.fft_len, self.hop_size)
        spec_ref = np.abs(stft_ref)**2
        self.chromafb = librosa.filters.chroma(self.fs, self.fft_len)
        raw_chroma_ref = np.dot(self.chromafb, spec_ref)
        self.chroma_ref = librosa.util.normalize(raw_chroma_ref, norm=2, axis=0)
        
        if self.chroma_info:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(self.chroma_ref, y_axis='chroma', x_axis='time')
            plt.colorbar()
            plt.title('Reference Chromagram')
            plt.tight_layout()
        
        # initialize arrays and matrices for live audio
        # double length of ref chroma for live chroma to make sure there is enough space w/out dynamically changing
        self.N = self.chroma_ref.shape[1] * 2  # rows are live
        self.M = self.chroma_ref.shape[1]      # cols are ref
        self.C = np.zeros((self.N, self.M))
        self.D = np.empty((self.N, self.M))
        self.D.fill(float('inf'))
        
        self.chroma_live = np.zeros((12, self.N))
        
        # TODO: use pyaudio buffer
        self.buf = []
        self.path = []
        
        # pointers & variables for LiveNote algorithm
        self.cost_matrix_ptr = 0
        self.ref_ptr = 0
        self.live_ptr = 0
        self.previous = None
        self.run_count = 0
        
        if self.chroma_info:
            self.chroma_ptr = 0
        
    def insert(self, live_audio_buf):
        # store incoming music
        self.buf += live_audio_buf
        
        # dealing with out of bounds issue
        if self.ref_ptr >= self.M - 1 or self.live_ptr >= self.N - 1:
            return "stop"
        
        # LiveNote algorithm:
        # identify window for one chroma col
        while len(self.buf) >= self.fft_len:
            section = np.array(self.buf[:self.fft_len])
            self.buf = self.buf[self.hop_size:]
            win = section * np.hanning(len(section))
            chroma = librosa.feature.chroma_stft(y=win, sr=self.fs, n_fft=self.fft_len, hop_length=self.hop_size)
            
            dft = np.fft.rfft(win)
            spec = np.abs(dft)**2
            raw_chroma = np.dot(self.chromafb, spec)
            chroma = librosa.util.normalize(raw_chroma, norm=2, axis=0)
            
            if self.chroma_info:
                self.chroma_live[:, self.chroma_ptr] = chroma
                self.chroma_ptr += 1

            # update cost matrix
            self.update_cost_matrix(chroma)
            
            # update distance (ie total path cost) matrix
            while (self.ref_ptr < self.cost_matrix_ptr - 1) and (self.live_ptr < self.N - 1):
                inc = self.get_inc()
                
                if inc != "column":
                    self.live_ptr += 1
                    for k in range(self.ref_ptr - self.search_band_width + 1, self.ref_ptr + 1):
                        if k > 0:
                            self.update_distance_matrix(self.live_ptr, k)
                
                if inc != "row":
                    self.ref_ptr += 1
                    for k in range(self.live_ptr - self.search_band_width + 1, self.live_ptr + 1):
                        if k > 0:
                            self.update_distance_matrix(k, self.ref_ptr)

                if inc == self.previous:
                    self.run_count += 1
                else:
                    self.run_count = 1
                
                if inc != "both":
                    self.previous = inc

                # update path
                self.path.append((self.ref_ptr, self.live_ptr))
    
    ########################
    ##  Helper functions  ##
    ########################
    
    def update_cost_matrix(self, chroma):
        ''' 
        Assumes chroma is just one column.
        Adds new row to matrix.
        '''
        
        cost = np.empty((1, self.M))
        for i in range(self.M):
            cost[0, i] = 1 - np.true_divide(np.dot(self.chroma_ref[:, i], chroma), (np.linalg.norm(self.chroma_ref[:, i]) * np.linalg.norm(chroma)))

        self.C[self.cost_matrix_ptr, :] = cost[0, :]

        # if first time adding to cost matrix, also initialize distance matrix
        if self.cost_matrix_ptr == 0:
            self.D[0, 0] = self.C[0, 0]

        self.cost_matrix_ptr += 1
        
    def get_inc(self):
        if self.live_ptr < self.search_band_width:
            return "both"

        if self.run_count > self.max_run_count:
            if self.previous == "row":
                return "column"
            else:
                return "row"
        
        y1 = self.live_ptr
        x1 = self.ref_ptr - self.search_band_width + np.argmin(self.D[y1, self.ref_ptr - self.search_band_width : self.ref_ptr + 1])
        x2 = self.ref_ptr
        y2 = self.live_ptr - self.search_band_width + np.argmin(self.D[self.live_ptr - self.search_band_width : self.live_ptr + 1, x2])
        (x, y) = (x1, y1) if (self.D[y1, x1] < self.D[y2, x2]) else (x2, y2)
        
        if x < self.ref_ptr:
            return "row"
        elif y < self.live_ptr:
            return "column"
        return "both"

    def update_distance_matrix(self, i, j):
        costs = []
        if i > 0:
            costs.append(self.D[i-1, j])
        if j > 0:
            costs.append(self.D[i, j-i])
        if i > 0 and j > 0:
            costs.append(self.D[i-1, j-1])
        
        if costs != []:
            cost = min(costs)
            self.D[i, j] = cost + self.C[i, j]
        
        # else, self.D[i, j] will remain 'inf'
        
    def stft(self, x, fft_len, hop_size):
        L = fft_len
        H = hop_size

        # use centered window by zero-padding
        x = np.concatenate((np.zeros(L/2), x))

        N = len(x)

        num_bins = 1 + L/2
        num_hops = int(((N - L)/H) + 1)

        stft = np.empty((num_bins, num_hops), dtype=complex)

        M = num_hops
        if self.chroma_info:
            print "Calculating dft for", M, "hops."
        
        for m in range(M):
            section = x[(m*H):((m*H) + L)]
            win = section * np.hanning(len(section))
            stft[:, m]= np.fft.rfft(win)

        return stft

class test_single_recording_livenote():
    
    def __init__(self, ref_recording, live_recording, ref_ground_truth, live_ground_truth, params, debug_params):
        
        self.dtw = LiveNote(ref_recording, params, debug_params)  
        self.live_recording, fs = librosa.load(live_recording)
        assert(fs == 22050)

        self.chroma_info = debug_params['chroma']
        self.song_info = debug_params['song']
        self.error_info = debug_params['error']
        self.error_detail = debug_params['error_detail']
        
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
        '''Evaluate single piece of music with LiveNote.'''
        # Emulate live recording via creation of buffers
        buffers = np.array_split(self.live_recording, buf_size)
        # For each buffer, get the synchronization estimate (ie the estimated position)
        # via call to insert
        for buf in buffers:
            est = self.dtw.insert(buf.tolist())
            if est == "stop":
                break
        
        self.sync_ests = self.dtw.path
        
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
        self.error = self.get_error()
    
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