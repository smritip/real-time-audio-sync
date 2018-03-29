import numpy as np

# Implementation of Online Time Warping Based on Dixon 2005
# This follows the algorithm in the paper very closely, but also fixes some bugs
class OnlineTimeWarping(object):
    def __init__(self, ref, params):
        super(OnlineTimeWarping, self).__init__()

        self.c = params['c']
        self.max_run_count = params['max_run_count']

        F = ref.shape[0]     # number of features
        N = ref.shape[1]     # length of ref sequence
        M = 2 * ref.shape[1] # length of live sequence (2x for now)

        # refernce sequence
        self.ref = ref 
        
        # live sequence:
        self.live = -1 * np.ones((F, M))
        
        # cost matrix
        self.cost = -1 * np.ones((M,N))
        
        # acc cost matrix. Initialize with large values to handle
        # non-computed edges
        self.acc_cost = 10**10 * np.ones((M,N))
                
        # algorithm state vars
        self.t = 0 # index into live
        self.j = 0  # index into ref
        self.previous = None
        self.run_count = 1
        self.direction = "Both"
        self.path = []
        self.first_insert = True
        
    def insert(self, live_sample):
        # print '\n\nInsert: cur index', self.t, self.j

        if self.first_insert:
            self.first_insert = False
            self.live[:,self.t] = live_sample
            self.eval_path_cost(self.t, self.j)
            return
        
        # Process a Row ('Row' or 'Both')
        assert(self.direction == 'Row' or self.direction == 'Both')

        self.t += 1
        
        # check end condition
        if self.t >= self.live.shape[1]:
            print 'Done. Ran out of room in pre-allocated live-sequence'
            return
        self.live[:,self.t] = live_sample
        
        k1 = max(0, self.j - self.c + 1)
        k2 = self.j + 1
        # print 'row[{} : {},{}]'.format(self.t, k1, k2)
        for k in range(k1, k2):
            self.eval_path_cost(self.t, k)

        while True:
            # Process a Column ('Column' or 'Both')
            if self.direction != 'Row':
                self.j += 1
                # check end conditions
                if self.j >= self.ref.shape[1]:
                    print 'Done. Ran out of ref-sequence'
                    return "stop"
                
                k1 = max(0, self.t - self.c + 1)
                k2 = self.t + 1
                # print 'col[{},{} : {}]'.format(k1, k2, self.j)
                for k in range(k1, k2):
                    self.eval_path_cost(k, self.j)            

            # Unlike paper, only call set_direction once for each loop.
            self.set_direction()
            # print self.direction
            
            # if we need to process a row next, we break and wait for next input
            if self.direction != 'Column':
                break
                
    
    # this function sets the entire "live" data at once, which is not how
    # it will run in real-time, but more closely matches the algorithm in the
    # paper.
    def set_live(self, live):
        self.t = 0
        self.j = 0
        self.previous = None
        self.direction = "Both"
        self.run_count = 1
        self.path = []
        
        self.fill_input(live)
        self.eval_path_cost(self.t, self.j)
        
        while True:
            # print '\n\ncur index', self.t, self.j
            
            # Unlike paper, only call set_direction once at the start of each loop.
            self.set_direction()
            # print self.direction
            
            # Process a Row ('Row' or 'Both')
            if self.direction != 'Column':
                self.t += 1
                # check end conditions
                if self.t >= live.shape[1]:
                    print 'Done. Ran out of live sequence'
                    break
                if self.t >= self.live.shape[1]:
                    print 'Done. Ran out of room in pre-allocated live-sequence'
                    break

                self.fill_input(live)
                k1 = max(0, self.j - self.c + 1)
                k2 = self.j + 1
                # print 'row[{} : {},{}]'.format(self.t, k1, k2)
                for k in range(k1, k2):
                    self.eval_path_cost(self.t, k)
                
            # Process a Column ('Column' or 'Both')
            if self.direction != 'Row':
                self.j += 1
                # check end conditions
                if self.j >= self.ref.shape[1]:
                    print 'Done. Ran out of ref-sequence'
                    break
                
                k1 = max(0, self.t - self.c + 1)
                k2 = self.t + 1
                # print 'col[{},{} : {}]'.format(k1, k2, self.j)
                for k in range(k1, k2):
                    self.eval_path_cost(k, self.j)
                
        # turn path into np.array at end.
        self.path = np.array(self.path)
    
    
    # this is INPUT(t). It fills in one additional step from the live input
    # into the current pointer, self.t
    def fill_input(self, live):
        self.live[:,self.t] = live[:,self.t] 
    
    # This does two things: calculates if next update show be 
    # row, column, or both. And finds the current optimal point and adds it to the
    # path.
    def set_direction(self):
        # print '  Set Direction at ({},{})'.format(self.t, self.j)

        # moved this calculation to happen all the time, so that we can
        # have a path right from the start:
        x,y = self.best_point()
        # print '    best_point: ({},{})'.format(x,y)
        self.path.append((x,y))

        # startup condition when we always alternate row/column
        if self.t < self.c:
            self.direction = 'Both'
        
        # slope enforcement: keep search space to within
        # a max runcount.
        elif self.run_count >= self.max_run_count:
            # print 'exceeded run count'
            self.direction = 'Column' if self.previous == 'Row' else 'Row'
        
        # choose row or column based on location of current optimal point
        elif x < self.t:
            self.direction = 'Column'
        elif y < self.j:
            self.direction = 'Row'
        else:
            self.direction = 'Both'
    
        # update run_count - the number of time we see
        # consecutive rows or consecutive coluns.
        if self.direction == self.previous:
            self.run_count += 1
        else:
            self.run_count = 1
        # print 'run-count', self.run_count
        if self.direction != 'Both':
            self.previous = self.direction
    
    
    # find minimum cost of current point (self.t, self.j)'s row and column
    def best_point(self):    
        # check row
        j1 = max(0, self.j - self.c + 1)
        j2 = self.j + 1
        # print '    row min [{}, {}:{}]'.format(self.t, j1, j2)
        best_j = j1 + np.argmin( self.acc_cost[self.t, j1:j2] )
        cost_j = self.acc_cost[self.t, best_j]
        
        # check column
        t1 = max(0, self.t - self.c + 1)
        t2 = self.t + 1
        # print '    col min [{}:{}, {}]'.format(t1, t2, self.j)
        best_t = t1 + np.argmin( self.acc_cost[t1:t2, self.j] )
        cost_t = self.acc_cost[best_t, self.j]
    
        # choose best cell from row or column:
        if cost_j < cost_t:
            return (self.t, best_j)
        else:
            return (best_t, self.j)
    
    
    # eval cost at location (x,y)
    def eval_path_cost(self, x, y):
        # print '  eval({},{})'.format(x,y)
        assert(self.live[0,x] != -1)
        
        # calculate cost at this cell:
        self.cost[x,y] = 1 - np.dot(self.live[:,x], self.ref[:,y])
        
        # initial condition for accumulated cost at (0,0)
        if x == 0 and y == 0:
            self.acc_cost[x,y] = self.cost[x,y]
            return
        
        # run one DTW step for updating accumulated cost at this cell
        steps = []
        if y > 0:
            steps.append( self.acc_cost[x,   y-1] + self.cost[x,y] )
        if x > 0:
            steps.append( self.acc_cost[x-1, y  ] + self.cost[x,y] )
        if x > 0 and y > 0:
            steps.append( self.acc_cost[x-1, y-1] + 2 * self.cost[x,y] )
                 
        best_cost = min(steps)
        # print '    ', steps
        # print '    ', best_cost
        self.acc_cost[x,y] = best_cost
        