import time
import array
from collections import defaultdict

print("Benchmarking accumulator...")
# Simulate 10,000 queries with 200 candidates each
t0 = time.time()
for q in range(10000):
    cand_data = {}
    # 5 channels hit
    for ch in range(5):
        for idx in range(ch*40, (ch+1)*40):
            c = cand_data.get(idx)
            if c is None:
                c = [10.0, 10.0, 0.0, 1 << ch, 1]
                cand_data[idx] = c
            else:
                c[0] += 10.0
                c[1] += 10.0
                if not (c[3] & (1 << ch)):
                    c[3] |= (1 << ch)
                    c[4] += 1
    # top 50 selection
    items = list(cand_data.items())
    items.sort(key=lambda x: x[1][0], reverse=True)
    top50 = items[:50]

elapsed = time.time() - t0
print(f"10,000 queries took {elapsed:.3f}s -> {10000/elapsed:,.0f} queries/sec")
