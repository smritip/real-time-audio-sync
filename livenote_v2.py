import numpy as np

class LiveNoteV2():
    '''
    Same implementation of LiveNote but does not allow addition of points in the past to the path.
    '''
    
    def __init__(self, ref, params, debug_params):
        
        # algorithm params
        self.search_band_width = params['search_band_width']  # max lookback
        self.max_run_count = params['max_run_count']  # max slope
        
        self.seq_ref = ref
        
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
        
        self.buf = np.zeros((self.F, self.N))
        self.path = []
        
        # pointers & variables for LiveNote algorithm
        self.ref_ptr = 0
        self.live_ptr = 0
        self.previous = None
        self.run_count = 0
        self.first_insert = True
        self.direction = "both"

    # Insert one sample
    def insert(self, live_sample):

        # if first insert, add input and then eval
        if self.first_insert:
            self.seq_live[:, self.live_ptr] = live_sample
            self.eval_path_cost(self.live_ptr, self.ref_ptr)
            self.first_insert = False
            return

        # PROCESS ROW
        self.live_ptr += 1

        # check bounds
        if self.live_ptr >= self.N:
            print "done - oob live"
            return

        # add input
        self.seq_live[:, self.live_ptr] = live_sample

        if self.live_ptr >= self.N:
            print 'Done. Ran out of room in pre-allocated live-sequence'
            return
        # eval paths
        k1 = max(0, self.ref_ptr - self.search_band_width + 1)
        k2 = self.ref_ptr + 1
        for k in range(k1, k2):
            self.eval_path_cost(self.live_ptr, k)

        # PROCESS COLUMN (until we change direction)
        while True:

            if self.direction != "row":

                self.ref_ptr += 1

                # check bounds
                if self.ref_ptr >= self.M:
                    print "done - oob ref"
                    return "stop"

                # eval paths
                k1 = max(0, self.live_ptr - self.search_band_width + 1)
                k2 = self.live_ptr + 1
                for k in range(k1, k2):
                    self.eval_path_cost(k, self.ref_ptr)

            # get new direction
            self.direction = self.get_direction()

            # update state of alg
            if self.direction == self.previous:
                self.run_count += 1
            else:
                self.run_count = 1

            if self.direction != "both":
                self.previous = self.direction

            # check if direction changed
            if self.direction != "column":
                break
    
    # Essentially gives alg the whole live sequence
    # Not to be used in live setting; use 'insert' instead
    def set_live(self, live):
        self.fill_input(live)

        # update cost matrix and accumulated cost matrix
        self.eval_path_cost(self.live_ptr, self.ref_ptr)

        while True:
        #while self.ref_ptr < self.M and self.live_ptr < self.N: 
            
            direction = self.get_direction()
                
            # process a row
            if direction != "column":
                self.live_ptr += 1

                if self.live_ptr >= live.shape[1]:
                    print 'done - ran out of live sequence'
                    break
                if self.live_ptr >= self.N:
                    print 'done - ran out of room in pre-allocated live-sequence'
                    break

                self.fill_input(live)
                k1 = max(0, self.ref_ptr - self.search_band_width + 1)
                k2 = self.ref_ptr + 1
                for k in range(k1, k2):
                    self.eval_path_cost(self.live_ptr, k)
                    

            # process a column
            if direction != "row":
                self.ref_ptr += 1
                if self.ref_ptr >= self.M:
                    print 'done - ran out of ref-sequence'
                    break

                k1 = max(0, self.live_ptr - self.search_band_width + 1)
                k2 = self.live_ptr + 1
                for k in range(k1, k2):
                    self.eval_path_cost(k, self.ref_ptr)

            if direction == self.previous:
                self.run_count += 1
            else:
                self.run_count = 1

            if direction != "both":
                self.previous = direction

    def fill_input(self, live):
        self.seq_live[:,self.live_ptr] = live[:,self.live_ptr]
    
        
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
        
    def get_direction(self):
        # first get best point (moved from end of pseudocode)
        (x, y) = self.calc_best_point()

        # update path with best point, only if best point is in forward direction
        if self.path == [] or (x > self.path[-1][0] and y >= self.path[-1][1]):
            self.path.append((x, y))

        if self.live_ptr < self.search_band_width:
            return "both"
        
        if self.run_count >= self.max_run_count:
            if self.previous == "row":
                return "column"
            else:
                return "row"
        

        if x < self.live_ptr:
            return "column"

        elif y < self.ref_ptr:
            return "row"

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

    # previous attempt at limiting path from going backwards, but doesn't make sense (?) and better to deal with in get_direction:
    # if ref1 + (ref2 - 1) < self.M:
    #     best_ref = ref1 + (ref2 - 1)
    # else:
    #     best_ref = ref1
    # cost_ref = self.acc_cost[self.live_ptr, best_ref]
    # cur_ref = ref1
    # for i in range(ref2 - ref1):
    #     cost = self.acc_cost[self.live_ptr, cur_ref]
    #     # update best ref and cost if cost is lower and point is not behind last point in path
    #     if (cost < cost_ref) and (ref1 + cur_ref >= self.path[-1][1]):  # path is (live, ref)
    #         best_ref = ref1 + cur_ref
            # cost_ref = self.acc_cost[self.live_ptr, best_ref]
