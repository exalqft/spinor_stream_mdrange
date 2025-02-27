# Kokkos based stream benchmark with 4D views representing so-called Dirac spinors

## Compilation instructions

Example compilation scripts are provided in the `compilation` directory for different architectures.

## Run scripts and results

Results of runs as well as run scripts are provided in the `run_scripts_and_results` directory for different architectures.

In the run scripts, the array extents are roughly matched. The worst matching is for the 5D version of the benchmark.

CPU benchmarks are run on both one and two sockets (using `OMP_PLACES=cores OMP_PROC_BIND=close` to ensure thread pinning to one socket when fewer threads are used than there are CPU cores).

See [results](run_scripts_and_results/spinor_stream_mdrange.pdf) for result plots.
