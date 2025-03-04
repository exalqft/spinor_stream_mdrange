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

2) **6D view with static spin and colour extents, internal loop over these**: `spinor-stream-mdrange-SC-static-internal.cpp` implements

```c++
Kokkos::View<val_t****[Ns][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;
[...]
void perform_set(...){
  Kokkos::parallel_for(
    "set",
    Policy<rank>(make_repeated_sequence<rank>(0), 
                 {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, 
                 tiling),
    KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
    {
      for(int is = 0; is < Ns; ++is){
        for(int ic = 0; ic < Nc; ++ic){
          a.view(i,j,k,l,is,ic) = scalar;
        }
      }
    });
  Kokkos::fence();
}
```

3) **Array of 4D views with internal loops over spin and colour**: `spinor-stream-mdrange-SC-array-internal.cpp` implements

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

4) **Array of 4D views with external loops over spin and colour**: `spinor-stream-mdrange-SC-array-external.cpp` implements

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

## Tiling

The tiling is determined using the `get_tiling` function which takes a view as its single argument from which it determines the view dimensions and the dynamic rank.
In the case of the "6D view" we use all dimensions instead, see [here](https://github.com/kostrzewa/spinor_stream_mdrange/blob/dfa17404e862f782f3d8fc73d434f438d477192a/spinor-stream-mdrange-SC-static.cpp#L108)

On CPUs we do OpenMP parallelisation over the two outermost dimensions (independently of whether we are dealing with a 4D MDRangePolicy or a 6D one) and on GPUs we extract the recommended tile sizes from the policy and use the outermost 4.

```c++
template <typename V>    
auto    
get_tiling(const V view)    
{    
  constexpr auto rank = view.rank_dynamic();    
  // extract the dimensions from the view layout (assuming no striding)    
  const auto & dimensions = view.layout().dimension;    
  Kokkos::Array<std::size_t,rank> dims;    
  for(int i = 0; i < rank; ++i){    
    dims[i] = dimensions[i];    
  }    
  // extract the recommended tiling for this view from a "default" policy     
  const auto rec_tiling = Policy<rank>(make_repeated_sequence<rank>(0),dims).tile_size_recommended();    
      
  if constexpr (std::is_same_v<typename V::execution_space, Kokkos::DefaultHostExecutionSpace>){                                           
    // for OpenMP we parallelise over the two outermost (leftmost) dimensions and so the chunk size    
    // for the innermost dimensions corresponds to the view extents    
    return Kokkos::Array<std::size_t,rank>({1,1,view.extent(2),view.extent(3)});    
  } else {    
    // for GPUs we use the recommended tiling for now, we just need to convert it appropriately    
    // from "array_index_type"    
    // unfortunately the recommended tile size may exceed the maximum block size on GPUs     
    // for large ranks -> let's cap the tiling at 4 dims    
    constexpr auto max_rank = rank > 4 ? 4 : rank;    
    Kokkos::Array<std::size_t,max_rank> res;    
    for(int i = 0; i < max_rank; ++i){    
      res[i] = rec_tiling[i];    
    }    
    return res;    
  }    
} 
```

## Compilation instructions

Example compilation scripts are provided in the `compilation` directory for different architectures.

## Run scripts and results

Results of runs as well as run scripts are provided in the `run_scripts_and_results` directory for different architectures.

CPU benchmarks are run on both one and two sockets (using `OMP_PLACES=cores OMP_PROC_BIND=close` to ensure thread pinning to one socket when fewer threads are used than there are CPU cores).

See [results](run_scripts_and_results/spinor_stream_mdrange.pdf) for result plots.
