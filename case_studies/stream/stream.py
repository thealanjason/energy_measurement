import numpy as np
import sys
import time

N  = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000_000
iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
a = np.zeros(N)
b = np.ones(N)
c = np.ones(N)
t0 = time.perf_counter()
for _ in range(iterations):
    a = b + 3.0 * c
t1 = time.perf_counter()
print(f"N={N}, iters={iterations}, time={t1 - t0:.6f}s")
