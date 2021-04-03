
from scipy.fftpack import fft
import scipy.fftpack
import matplotlib.pyplot as plt
from mpi4py import MPI
import math
import numpy as np

pi = math.pi

# generate the time space (time * 2 * pi)
timeMin = -20
timeMax = 20
n_samples = 2**10
t = 2 * pi * np.linspace(timeMin, timeMax, n_samples)

# np.sin(F * t) * exp(-(t - t_shift)**2 / 2 / gain**2)
# F - frequency [Hz]
# t - time [cycles] - time * 2*pi
# t_shift - time shift relating 0 (<0 - right; >0 - left) [cycles]
# gain - decay rate: less gain - higher decay
y = 0 # zero signal
y += np.sin(1.0 * t) * np.exp(-(t - 0 * 2 * pi)**2 / 2 / 20**2) # 1st wawve packet
y += np.sin(3.0 * t) * np.exp(-(t - 5 * 2 * pi)**2 / 2 / 20**2) # 2nd wawve packet
y += np.sin(5.5 * t) * np.exp(-(t - 10 * 2 * pi)**2 / 2 /5**2)  # 3rd wawve packet
y += np.sin(4.0 * t) * np.exp(-(t - 7 * 2 * pi)**2 / 2 / 15**2) # 4th wawve packet (added)

wind_pos_start = -20.0 * 2 * pi
wind_pos_stop = 20.0 * 2 * pi
window_width = 2.0 * 2 * pi

def get_windowed(rank, n_proc, y = y, wind_pos_start = -20.0 * 2 * pi, wind_pos_stop = 20.0 * 2 * pi, window_width = 1.0 * 2 * pi):
    array = [] # array for all window positionns time-domain functions
    # for all window positions
    x_0 = wind_pos_start
    step = int(n_samples / n_proc)
    
    for i in range(step * rank, step * (rank+1), 1):
        wind_pos = x_0 + (t[1]-t[0]) * i
        
        window_function = np.exp(-(t - wind_pos)**2 / 2 / window_width**2) # define current window function
        y_window = y * window_function # find result of windowing
        array.append(y_window) # add windowed function into array
    return array

def get_specgram(rank, n_proc, wind_pos_start = -20.0 * 2 * pi, wind_pos_stop = 20.0 * 2 * pi, window_width = 1.0 * 2 * pi):
    # get windowed function
    array = get_windowed(rank = rank, n_proc = n_proc, y = y, wind_pos_start = wind_pos_start, wind_pos_stop = wind_pos_stop, window_width = window_width)
    spect = [] # array for all window positionns spectrum 
    arr = np.array(array)
    # for all window positions
    for i in range(arr.shape[0]):
        y_window = arr[i] # use current windowing result
        spectrum = scipy.fftpack.fft(y_window) # get spectrum
        spect.append(abs(spectrum[0:int(spectrum.shape[0] / 2 -1)])**2) # get one half of the spectrum
    spectrogram = np.array(spect)
    return spectrogram

comm = MPI.COMM_WORLD
n_proc = comm.Get_size() # processors
rank = comm.Get_rank() # current rank

if rank == 0:
    print('nprocs = ', n_proc)
    t0 = MPI.Wtime() # measure start time

spectrogram = get_specgram(rank = rank, n_proc = n_proc, wind_pos_start = wind_pos_start, wind_pos_stop = wind_pos_stop, window_width = window_width)
# X,Y = count(proc = rank, points_X = 1000, n_proc = n_proc, minX = 0, maxX = 4, steps = 500, m = 50)
spectrogram = comm.gather(spectrogram, root=0)

if rank == 0:
    a = []
    # get all parts of spectrogram
    for i in range(n_proc):
        a.append(np.rot90(spectrogram[i]))
    
    # spectrogram parts concatenation
    out = np.concatenate((a), axis=1)
    
    # total time of calculation
    totalTime = MPI.Wtime() - t0
    print(np.round((totalTime),4) * 1000, 'mSec')
    np.savez('subtask4/example_'+str(n_proc), totalTime = np.round((totalTime),4) * 1000)
    
    fig = plt.imshow(out)
    plt.xticks(np.linspace(0, n_samples-1, 9), np.round(np.linspace(-20, 20, 9),2), fontsize=14)
    plt.yticks(np.linspace(0, n_samples / 2 -1, 9), np.round(np.linspace( n_samples / 80, 0, 9),2), fontsize=14) # 1024/80 - scale
    plt.xlabel('t, cycles', fontsize=14)
    plt.ylabel('Frequency, arb. units', fontsize=14)
    plt.savefig('spectrum', dpi = 100)
    plt.close()
