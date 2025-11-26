import numpy as np
import sys
import time

N  = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3

A = np.random.rand(N, N)
B = np.random.rand(N, N)
C = np.zeros((N, N))
t0 = time.perf_counter()
for _ in range(iterations):
    C = A @ B
t1 = time.perf_counter()
print(f"N={N}, iters={iterations}, time={t1 - t0:.6f}s")