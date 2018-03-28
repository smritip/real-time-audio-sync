import numpy as np


# taken from Eran
def DTW(seq_a, seq_b):
    # matrix dimensions
    M = seq_a.shape[1]
    N = seq_b.shape[1]

    # create the cost matrix:
    cost = 1 - np.dot(seq_a.T, seq_b)
    
    # accumulated cost matrix:
    acc_cost = np.zeros((M,N))

    # backtracking matrix:
    back = np.empty((M,N), dtype=np.int)

    # initalize accumulated cost matrix:
    acc_cost[0,0] = cost[0,0]
    back[0,0] = 2
    
    for i in range(1,M):
        acc_cost[i,0] = cost[i,0] + acc_cost[i-1, 0]
        back[i,0] = 1
    for j in range(1,N):
        acc_cost[0,j] = cost[0,j] + acc_cost[0, j-1]
        back[0,j] = 0
    
    STEPS = [(0, -1), (-1, 0), (-1, -1)]
    # DTW algorithm
    for i in range(1,M):
        for j in range(1,N):
            # check 3 steps ending up at i,j:
            options = (acc_cost[i  ,j-1] + cost[i,j],
                       acc_cost[i-1,j]   + cost[i,j],
                       acc_cost[i-1,j-1] + 2*cost[i,j])
            best_step = np.argmin(options)
            acc_cost[i,j] = options[best_step]
            back[i,j] = best_step
    
    # backtrack for path compuation:
    i = M - 1
    j = N - 1
    path = [(i,j)]
    while i > 0 or j > 0:
        step = STEPS[ back[i,j] ]
        i += step[0]
        j += step[1]
        path.append( (i,j) )
    path.reverse()
    path = np.array(path)
    return cost, acc_cost, path