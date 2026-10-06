"""Process-wide numerical runtime defaults."""
import os

# Avoid oversubscribing small matrix operations in the OFDM baselines.
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_variable, '4')
