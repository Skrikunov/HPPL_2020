
from mpi4py import MPI
import math
import numpy as np

def calculate_integral(start_val, end_val, steps, n_proc, rank, err_Estim = False):
    x_0 = start_val
    x_1 = end_val
    step = x_1 / n_proc
    points_X = steps
    
    # if points_X / n_proc is not integer
    blocks_X = int(points_X / n_proc) # points of x axis
    remainder = points_X % n_proc # remainder for the last block
    add = 0 # addition of the last block
    if rank == n_proc-1: # if current process is last
        add = remainder # addition is equal to remainder
        
    # integral
    x = np.linspace(x_0 + step * rank, x_0 + step * (rank + 1), add + blocks_X) # small step
    y = np.exp(x) * np.cos(x) # function
    I_1n = np.trapz(y = y, x = x, axis = 0) # trapezoidal integration
    
    if err_Estim:
        # for error estimation
        x = np.linspace(x_0 + step * rank, x_0 + step * (rank + 1), add + int(steps / (2 * n_proc))) # long step
        y = np.exp(x) * np.cos(x) # function
        I_2n = np.trapz(y = y, x = x, axis = 0) # trapezoidal integration

        # we can estimate error using double calculation: I_1n, I_2n
        # error estimation: ERR <= |I_2n - I_1n| / 3
        err = abs(I_2n - I_1n) / 3

        # print('Integral value:  ', I_1n)
        # print('Error estimation:', err)
        return I_1n, err
    else:
        return I_1n

comm = MPI.COMM_WORLD
n_proc = comm.Get_size() # processors
rank = comm.Get_rank() # current rank

if rank == 0:
    print('nprocs = ', n_proc)
    t0 = MPI.Wtime() # measure start time
    
#n = comm.bcast(n, root=0)
integral_part = calculate_integral(start_val = 0, end_val = 2 * math.pi, steps = 10000000, n_proc = n_proc, rank = rank, err_Estim = False) # calculate integral part with current rank
integral = comm.reduce(integral_part, op = MPI.SUM, root = 0) # accumulate all values to get final result

if rank == 0:
    print ('integral = ', integral)
    totalTime =  MPI.Wtime() - t0
    print('time = ', np.round((totalTime),4) * 1000, 'mSec')

    np.savez('subtask3/example_'+str(n_proc), totalTime = np.round((totalTime),4) * 1000)
