class DTW_x():
    
    def __init__(self, ref_recording, params):
        '''Initialize DTW variant with reference recording and any params.'''
        self.ref_recording = ref_recording
        self.params = params
        self.pos = 0
        self.debug = False
        
    def insert(self, live_audio_buf):
        '''Return synchronization estimates.'''
        self.pos += len(live_audio_buf)
        return self.pos
    
    def set_debug():
        self.debug = True

class test_single_recording():
    
    def __init__(self, ref_recording, live_recording, ground_truth, params, dtw):
        self.dtw = dtw(ref_recording, params)  # passed in 'dtw', which is some form of DTW_x...
        self.live_recording = live_recording
        self.ground_truth = ground_truth        
        
    def evaluate(self, buf_size):
        '''Evaluate single piece of music with one DTW variant.'''
        # Emulate live recording via creation of buffers
        buffers = np.array_split(self.live_recording, buf_size)
        
        # For each buffer, get the synchronization estimate (ie the estimated position)
        # via call to insert
        sync_ests = []
        for buf in buffers:
            est = self.dtw.insert(buf)
            sync_ests.append(est)
        
        # Compare estimates to ground truth, and return score
        score = sync_ests * self.ground_truth
        return score

class test_DTW():  # multiple songs, 1 DTW_x algorithm
    
    def __init__(self, ref_recordings, live_recordings, ground_truths, params, dtw):
        self.ref_recordings = ref_recordings
        self.live_recordings = live_recordings
        self.ground_truths = ground_truths
        self.params = params
        self.dtw = dtw
        
    def evaluate(self, buf_size):
        '''Evaluate a DTW variant (test with several pieces).'''
        scores = []
        for ref in self.ref_recordings:
            for i in range(len(self.live_recordings)):
                live = self.live_recordings[i]
                truth = self.ground_truths[i]
                test = test_single_recording(ref, live, truth, self.params, self.dtw)
                score = test.evaluate(buf_size)
                scores.append(score)
                
        scores = np.array(scores)
        
        return np.mean(scores)

class test_all():  # multiple songs, multiple DTWs (test each DTW with multiple recordings)
    
    def __init__(self, ref_recordings, live_recordings, ground_truths, params, dtws):
        self.ref_recordings = ref_recordings
        self.live_recordings = live_recordings
        self.ground_truths = ground_truths
        self.params = params
        self.dtws = dtws
        
    def evaluate(self):
        '''Evaluate all DTW variants (with all pieces).'''
        scores = []
        for dtw in dtws:
            test = test_DTW(self.ref_recordings, self.live_recordings, self.ground_truths, self.params, dtw)
            score = test.evaluate()
            scores.append(score)
            
        return scores