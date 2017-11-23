class OTW():
    
    def __init__(self, ref_recording, params):
        # reference audio, fs = 22050
        self.ref, self.fs = librosa.load(ref_recording)
        
        # TODO: add debug to params
        self.fft_len = params['fft_len']
        self.hop_size = params['hop_size']
        self.search_band_width = params['search_band_width']
        self.max_run_count = params['max_run_count']  # max slope
        
        # create STFT and chromagram of reference audio
        # TODO: check which stft and chroma to do (additions: tuning, cens, etc)
        # TODO: which librosa functions to use; chroma here seems weird
        # TODO: do I need to calculate STFT at all ?
        # stft_ref = librosa.stft(ref, fft_len, hop_size)
        self.chroma_ref = librosa.feature.chroma_stft(y=self.ref, sr=self.fs, n_fft=self.fft_len, hop_length=self.hop_size)
        #plt.imshow(self.chroma_ref, origin="lower", aspect='auto', cmap='Greys');
        #plt.colorbar();
        
        # initialize arrays and matrices for live audio
        # double length of ref STFT and chroma to make sure there is enough space w/out dynamically changing
        # TODO: do I need to calculate STFT at all ?
        # stft_live = np.empty((stft_ref.shape[0], stft_ref.shape[1]*2))
        # TODO : do I need chroma for live at all?
        # self.chroma_live = np.zeros((chroma_ref.shape[0], chroma_ref.shape[1]*2))
        
        self.N = self.chroma_ref.shape[1] * 2  # rows are live
        self.M = self.chroma_ref.shape[1]      # cols are ref
        self.C = np.zeros((self.N, self.M))
        self.D = np.empty((self.N, self.M))
        self.D.fill(float('inf'))
        
        # TODO: use pyaudio buffer
        self.buf = []
        self.path = []
        
        # useful pointers
        self.cost_matrix_ptr = 0
        self.ref_ptr = 0
        self.live_ptr = 0
        self.previous = None
        self.run_count = 0
        
    def insert(self, live_audio_buf):
        # store incoming music
        self.buf += live_audio_buf
        
        # TODO: deal with out of bounds issue...
        if self.ref_ptr >= self.M - 1 or self.live_ptr >= self.N - 1:
            return "stop"
        
        # OTW algorithm:
        # identify window for one chroma col
        while len(self.buf) >= self.fft_len:
            win = np.array(self.buf[:self.fft_len])
            self.buf = self.buf[self.hop_size:]
            chroma = librosa.feature.chroma_stft(y=win, sr=self.fs, n_fft=self.fft_len, hop_length=self.hop_size)

            # update cost matrix
            # TODO: why is this more than one column?? (fix: sent first element...)
            self.update_cost_matrix(chroma[:, 0])
            
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
        #if i > 0 and (i-1) < self.N and j < self.M:
            costs.append(self.D[i-1, j])
        if j > 0:
        #if j > 0 and (j-1) < self.M and i < self.N:
            costs.append(self.D[i, j-i])
        if i > 0 and j > 0:
        #if i > 0 and j > 0 and (i-1) < self.N and (j-1) < self.M:
            costs.append(self.D[i-1, j-1])
        
        if costs != []:
        #if costs != [] and i < self.N and j < self.M: 
            cost = min(costs)
            self.D[i, j] = cost + self.C[i, j]
        
        # else, self.D[i, j] will remain 'inf'

class test_single_recording():
    
    def __init__(self, ref_recording, live_recording, ref_ground_truth, live_ground_truth, params):
        # TODO: pass in 'dtw', which is some form of DTW_x...
        self.dtw = OTW(ref_recording, params)  
        self.live_recording, fs = librosa.load(live_recording)
        
        self.ref_ground_truth_time = []
        self.ref_ground_truth_beats = []
        self.live_ground_truth_time = []
        self.live_ground_truth_beats = []
        
        ref_song = ref_recording[:-4]
        ref_csv_file = ref_song + '.csv'
        live_song = live_recording[:-4]
        live_csv_file = live_song + '.csv'
     
        with open(ref_csv_file) as ref_csv_data:
            reader = csv.reader(ref_csv_data)
            for row in reader:
                self.ref_ground_truth_time.append(float(row[0]))
                self.ref_ground_truth_beats.append(float(row[1]))
                
        with open(live_csv_file) as live_csv_data:
            reader = csv.reader(live_csv_data)
            for row in reader:
                self.live_ground_truth_time.append(float(row[0]))
                self.live_ground_truth_beats.append(float(row[1]))
        
    def evaluate(self, buf_size):
        '''Evaluate single piece of music with one DTW variant.'''
        # Emulate live recording via creation of buffers
        buffers = np.array_split(self.live_recording, buf_size)
        ctr = 0
        # For each buffer, get the synchronization estimate (ie the estimated position)
        # via call to insert
        for buf in buffers:
            if ctr%100 == 0:
                print ctr
            est = self.dtw.insert(buf.tolist())
            if est == "stop":
                break
            ctr += 1
        
        self.sync_ests = self.dtw.path
        
        # Compare estimates to ground truth, and return error
        error = self.get_error()
        return error
    
    def get_error(self):
        # TODO: determine better way to generate score
        error = 0
        for (l, r) in self.sync_ests:
            l_beat = self.get_beat(l, self.live_ground_truth_time, self.live_ground_truth_beats)
            r_beat = self.get_beat(r, self.ref_ground_truth_time, self.ref_ground_truth_beats)
            #diff = (r_beat - l_beat)**2
            diff = r_beat - l_beat
            error += diff
        return error
        
    def get_beat(self, t, gtime, gbeats):
        '''Given a point and set of ground truths, determine which beat it was closest to.''' 
        ff = float(self.dtw.fs) / self.dtw.hop_size
        gsam = [x * ff for x in gtime]
        for i in range(len(gsam) - 1):
            if gsam[i] <= t < gsam[i+1]:
                return gbeats[i]
        # TODO: find better alternative return value
        return 0
        