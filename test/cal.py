import numpy as np

data = np.loadtxt(r"C:\Users\lab29\Downloads\test\0.0K-1.0M_2.0M-5k-2025-08-19_17-28-50.dat", dtype=float, delimiter=',')
np.set_printoptions(threshold=np.inf)
print(np.diff(data[:,0]))