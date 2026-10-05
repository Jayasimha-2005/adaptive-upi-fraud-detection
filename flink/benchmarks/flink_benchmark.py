import time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flink.batch.historical_pipeline import main

start = time.perf_counter()
main()
print(f"Elapsed seconds: {time.perf_counter() - start:.3f}")
