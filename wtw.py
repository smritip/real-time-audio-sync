class WTW():
    
    def __init__(self, ref_recording, params, debug=False):
        # reference audio, fs = 22050
        self.ref, self.fs = librosa.load(ref_recording)
        
        # params
        self.fft_len = params['fft_len']
        self.hop_size = params['hop_size']
        self.dtw_win_size = params['dtw_win_size']
        self.dtw_hop_size = params['dtw_hop_size']
        
        self.debug = debug
        
        # create STFT, spectrogram, and chromagram of reference audio
        # TODO: check additions (tuning, normalization, etc.)
        # note: using own stft function (vs. librosa's) b/c insert will also use self.stft
        stft_ref = self.stft(self.ref, self.fft_len, self.hop_size)
        spec_ref = np.abs(stft_ref)**2
        self.chromafb = librosa.filters.chroma(self.fs, self.fft_len)
        raw_chroma_ref = np.dot(self.chromafb, spec_ref)
        self.chroma_ref = librosa.util.normalize(raw_chroma_ref, norm=2, axis=0)
        
        if self.debug:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(self.chroma_ref, y_axis='chroma', x_axis='time')
            plt.colorbar()
            plt.title('Reference Chromagram')
            plt.tight_layout()
        
        # initialize arrays and matrices for live audio
        # double length of ref chroma for live chroma to make sure there is enough space w/out dynamically changing
        self.N = self.chroma_ref.shape[1] * 2  # rows are live
        self.M = self.chroma_ref.shape[1]      # cols are ref
        
        self.chroma_live = np.zeros((12, self.N))
        
        # TODO: use pyaudio buffer
        self.buf = []
        self.path = []
        
        # pointers & variables for OTW algorithm
        #self.cost_matrix_ptr = 0
        #self.ref_ptr = 0
        #self.live_ptr = 0
        #self.previous = None
        #self.run_count = 0
        
        self.chroma_ptr = 0
        self.live_ptr = 0
        self.ref_ptr = 0
        self.windows = []
        
    def insert(self, live_audio_buf):
        # store incoming music
        self.buf += live_audio_buf
        
        # dealing with out of bounds issue
        if self.ref_ptr >= self.M - 1 or self.live_ptr >= self.N - 1:
            return "stop"
        
        # WTW algorithm:
        # add chroma col
        while len(self.buf) >= self.fft_len:
            section = np.array(self.buf[:self.fft_len])
            self.buf = self.buf[self.hop_size:]
            win = section * np.hanning(len(section))
            chroma = librosa.feature.chroma_stft(y=win, sr=self.fs, n_fft=self.fft_len, hop_length=self.hop_size)
            
            dft = np.fft.rfft(win)
            spec = np.abs(dft)**2
            raw_chroma = np.dot(self.chromafb, spec)
            chroma = librosa.util.normalize(raw_chroma, norm=2, axis=0)
            
            self.chroma_live[:, self.chroma_ptr] = chroma
            self.chroma_ptr += 1

            # boundary conditions:
            if self.ref_ptr >= (self.M - 1 - self.dtw_win_size) or self.live_ptr >= (self.N - 1 - self.dtw_win_size):
                return "stop"

            # perform DTW on WTW window, if possible
            while (self.chroma_ptr - self.live_ptr >= self.dtw_win_size):
                chroma_x = self.chroma_ref[:, self.live_ptr + self.dtw_win_size]
                chroma_y = self.chroma_ref[:, self.ref_ptr + self.dtw_win_size]
                cost_xy = self.get_cost_matrix(chroma_x, chroma_y)
                D, B = self.run_dtw(cost_xy)
                #self.windows.append((D, self.live_ptr))
                subpath = find_path(B)
                next_start = self.live_ptr + self.dtw_hop_size
                for i in range(len(subpath)):
                    l = subpath[i][0]
                    r = subpath[i][1]
                    # TODO: <= vs just <
                    if l <= next_start:
                        self.path.append((l, r))
                    elif l > next_start:
                        self.live_ptr = subpath[i-1][0]
                        self.ref_ptr = subpath[i-1][1]
                        break
                

    def debug_on(self):
        self.debug = True
        
    def debug_off(self):
        self.debug = False
    
    ########################
    ##  Helper functions  ##
    ########################
        
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
        if self.debug:
            print "Calculating dft for", M, "hops."
        
        for m in range(M):
            section = x[(m*H):((m*H) + L)]
            win = section * np.hanning(len(section))
            stft[:, m]= np.fft.rfft(win)

        return stft

    def get_cost_matrix(self, x, y) :
        N = x.shape[1]
        M = y.shape[1]
        max_range= max(N, M)
        cost = np.empty((N, M))
        for i in range(N):
            for j in range(M):
                cost[i, j] = 1 - np.true_divide(np.dot(x[:, i], y[:, j]), (np.linalg.norm(x[:, i]) * np.linalg.norm(y[:, j])))
        
        return cost

    def run_dtw(self, C):
        n = C.shape[0]
        m = C.shape[1]
        D = np.empty((n, m))
        
        # each entry in B will be like a "pointer" to the point it came from
        # (0 = origin, 1 = from left, 2 = from diagonal, 3 = from below)
        B = np.empty((n, m))
        
        # initialize origin
        D[0, 0] = C[0, 0]
        B[0, 0] = 0
        
        # initialize first column
        cost = C[0, 0]
        for i in range(1, n):
            cost += C[i, 0]
            D[i, 0] = cost
            B[i, 0] = 3
        
        # initialize first row
        cost = C[0, 0]
        for i in range(1, m):
            cost += C[0, i]
            D[0, i] = cost
            B[0, i] = 1
        
        # calculate accumulated cost for rest of matrix
        for i in range(1, n):
            for j in range(1, m):
                p_costs = [(i-1, j), (i, j-1), (i-1, j-1)]
                min_cost = D[p_costs[0][0], p_costs[0][1]]
                min_indices = p_costs[0]
                for k in range(1, len(p_costs)):
                    c = D[p_costs[k][0], p_costs[k][1]]
                    if c < min_cost:
                        min_cost = D[p_costs[k][0], p_costs[k][1]]
                        min_indices = p_costs[k]
            
                D[i, j] = min_cost + C[i, j]
                
                ptr = {(i-1, j): 3, (i, j-1): 1, (i-1, j-1): 2}
                B[i, j] = ptr[min_indices]
                
        return D, B

    def find_path(self, B) :
        n = B.shape[0]
        m = B.shape[1]
        current = (n-1, m-1)
        path = [current]
        goal = (0, 0)
        while current != goal:
            ptr = B[current[0], current[1]]

            if ptr == 1:  # go left
                next_pt = (current[0], current[1] - 1)    
            elif ptr == 2:  # go diagonal
                next_pt = (current[0] - 1, current[1] - 1)
            elif ptr == 3:  # go down
                next_pt = (current[0] - 1, current[1])
                
            path.append(next_pt)
            current = next_pt
        
        path.reverse()
            
        return path