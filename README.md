# Kokkos based stream benchmark with 4D views representing so-called Dirac spinors

This benchmark implements the five stream operations `set, copy, scale, add` and `triad` on complex-valued six-dimensional objects called **Dirac Spinors** which have 4 large dimensions representing space-time (sized between 16^4 and 64^4 per node in a typical calculation) and two small extents, "spin" and "colour" of size 4 and 3, respectively.

In lattice quantum field theory calculations we deal with even higher-dimensional objects in the construction of various observables and the performance of level 1 linear algebra operations as well as reductions over these objects (or slices thereof) are potential bottlenecks.

Three implementations are considered:

1) **6D view with static spin and colour extents**: `spinor-stream-mdrange-SC-static.cpp` implements

```c++
Kokkos::View<val_t****[Ns][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;
[...]
void perform_set(...){
  Kokkos::parallel_for(
    "set",
    Policy<rank>(make_repeated_sequence<rank>(0), 
                 {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}, 
                 tiling),
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                  const StreamIndex is, const StreamIndex ic)
    { 
      a.view(i,j,k,l,is,ic) = scalar; 
    });
  Kokkos::fence();
}
```

2) **Array of 4D views with internal loops over spin and colour**: `spinor-stream-mdrange-SC-array-internal.cpp` implements

```c++
Kokkos::View<val_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;
Kokkos::Array<Kokkos::Array<StreamDeviceArray,Nc>,Ns> view;
[...]
void perform_set(...){
  Kokkos::parallel_for(
    "set", 
    Policy<rank>(make_repeated_sequence<rank>(0), make_repeated_sequence<rank>(stream_array_size), tiling),
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
    { 
      for(int is = 0; is < Ns; ++is){
        for(int ic = 0; ic < Nc; ++ic){
          a.view[is][ic](i,j,k,l) = scalar; 
        }
      }
  });
  Kokkos::fence();
}
```

3) **Array of 4D views with external loops over spin and colour**: `spinor-stream-mdrange-SC-array-external.cpp` implements

```c++
Kokkos::View<val_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;
Kokkos::Array<Kokkos::Array<StreamDeviceArray,Nc>,Ns> view;
[...]
void perform_set(...){
  for(int is = 0; is < Ns; is++){
    for(int ic = 0; ic < Nc; ic++){
      Kokkos::parallel_for(
          "set", 
          Policy<rank>(make_repeated_sequence<rank>(0), make_repeated_sequence<rank>(stream_array_size),tiling),
          KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
          { 
            a.view[is][ic](i,j,k,l) = scalar; 
          });
    }
  }
  Kokkos::fence();
}
```

The "array of views" approach on the one hand alleviates the need for views with higher dimensions than 8 and additionally, it performs better on GPUs.
Unfortunately, on CPUs it performs much worse when the working set fits into the last-level-cache.

The "external" loop probably simplifies autovectorisation and also leads to better performance on GPUs compared to the "internal" approach.
On CPUs, however, it is slower than the "internal" approach.

## Compilation instructions

Example compilation scripts are provided in the `compilation` directory for different architectures.

## Run scripts and results

Results of runs as well as run scripts are provided in the `run_scripts_and_results` directory for different architectures.

CPU benchmarks are run on both one and two sockets (using `OMP_PLACES=cores OMP_PROC_BIND=close` to ensure thread pinning to one socket when fewer threads are used than there are CPU cores).

See [results](run_scripts_and_results/spinor_stream_mdrange.pdf) for result plots.
