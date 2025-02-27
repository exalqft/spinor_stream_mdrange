/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
//
// Rephrasing as a benchmark for spinor linear algebra by Bartosz Kostrzewa (Uni Bonn) 
//
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include <utility>
#include <iostream>
#include <limits>

#include <sys/time.h>


#define STREAM_NTIMES 20

using val_t = Kokkos::complex<double>;
constexpr val_t ainit(1.0, 0.1);
constexpr val_t binit(1.1, 0.2);
constexpr val_t cinit(1.3, 0.3);

//using val_t = double;
//constexpr val_t ainit(1.0);
//constexpr val_t binit(1.1);
//constexpr val_t cinit(1.3);

#define HLINE "-------------------------------------------------------------\n"

template <int Ns, int Nc>
using StreamDeviceArray =
    Kokkos::View<val_t****[Ns][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;
#if defined(KOKKOS_ENABLE_CUDA)
template <int Ns, int Nc>
using constStreamDeviceArray =
    Kokkos::View<const val_t ****[Ns][Nc], Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#else
template <int Ns, int Nc>
using constStreamDeviceArray =
    Kokkos::View<const val_t ****[Ns][Nc], Kokkos::MemoryTraits<Kokkos::Restrict>>;
#endif
template <int Ns, int Nc>
using StreamHostArray = typename StreamDeviceArray<Ns,Nc>::HostMirror;

using StreamIndex = long int;

template <int rank>
using Policy      = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

template <typename V, typename P>
auto
get_tiling(const V view, const P policy)
{
  if constexpr (std::is_same_v<typename V::execution_space, Kokkos::DefaultHostExecutionSpace>){
    // for OpenMP we parallelise over the outermost (leftmost) dimensions
    return Kokkos::Array<std::size_t,view.rank()>({1,1,view.extent(2),view.extent(3),view.extent(4),view.extent(5)});
  } else {
    // for GPUs we use the recommended tiling for now, we just need to convert it appropriately
    // from "array_index_type"
    // oh my.. the recommended tile size exceeds the maximum block size on GPUs for large ranks
    // let's cap the tiling at 4 dims
    constexpr auto max_rank = view.rank() > 4 ? 4 : view.rank();
    Kokkos::Array<std::size_t,max_rank> res;
    const auto rec = policy.tile_size_recommended();
    for(int i = 0; i < max_rank; ++i){
      res[i] = rec[i];
    }
    return res;
  }
}

template <std::size_t... Idcs>
constexpr Kokkos::Array<std::size_t, sizeof...(Idcs)>
make_repeated_sequence_impl(std::size_t value, std::integer_sequence<std::size_t, Idcs...>)
{
  return { ((void)Idcs, value)... };
}

template <std::size_t N>
constexpr Kokkos::Array<std::size_t,N> 
make_repeated_sequence(std::size_t value)
{
  return make_repeated_sequence_impl(value, std::make_index_sequence<N>{});
}

template <int Ns, int Nc>
struct deviceSpinor {
  deviceSpinor() = delete;

  deviceSpinor(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init)
  {
    do_init(N0,N1,N2,N3,view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, 
          StreamDeviceArray<Ns,Nc> & V, const val_t init){
    Kokkos::realloc(Kokkos::WithoutInitializing, V, N0, N1, N2, N3);
    
    // need a const view to get the constexpr rank
    const StreamDeviceArray<Ns,Nc> vconst = V;
    constexpr auto rank = vconst.rank();
    const Policy<rank> default_policy(make_repeated_sequence<rank>(0), {N0,N1,N2,N3,Ns,Nc});
    const auto tiling = get_tiling(vconst, default_policy);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), {N0,N1,N2,N3,Ns,Nc}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                    const StreamIndex is, const StreamIndex ic)
      { 
        V(i,j,k,l,is,ic) = init; 
      }
    );
    Kokkos::fence();
  }

  StreamDeviceArray<Ns,Nc> view;
};

template <int Ns, int Nc>
struct constDeviceSpinor {
  constStreamDeviceArray<Ns,Nc> view;
};

template <int Ns, int Nc>
struct hostSpinor {
  StreamHostArray<Ns,Nc> view;
};


int parse_args(int argc, char **argv, StreamIndex &stream_array_size) {
  // Defaults
  stream_array_size = 32;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Create stream views containing [Ns][Nc]<N>^4 elements.\n"
      "     Default: 32\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"nelements", required_argument, NULL, 'n'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "n:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 'n': stream_array_size = atoi(optarg); break;
      case 'h':
        printf("%s", help_string.c_str());
        return -2;
        break;
      case 0: break;
      default:
        printf("%s", help_string.c_str());
        return -1;
        break;
    }
  return 0;
}

template <int Ns, int Nc>
void perform_set(const deviceSpinor<Ns,Nc> a, const val_t scalar) {
  constexpr auto rank = a.view.rank();
  const auto stream_array_size = a.view.extent(0);
  const Policy<rank> 
    default_policy(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}); 
  const auto tiling = get_tiling(a.view, default_policy);
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

template <int Ns, int Nc>
void perform_copy(const deviceSpinor<Ns,Nc> a, const deviceSpinor<Ns,Nc> b) {
  constexpr auto rank = a.view.rank();
  const auto stream_array_size = a.view.extent(0);
  const Policy<rank> 
    default_policy(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}); 
  const auto tiling = get_tiling(a.view, default_policy);
  Kokkos::parallel_for(
      "copy",
      Policy<rank>(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}, 
                   tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                    const StreamIndex is, const StreamIndex ic)
      { 
        b.view(i,j,k,l,is,ic) = a.view(i,j,k,l,is,ic);
      });
  Kokkos::fence();
}

template <int Ns, int Nc>
void perform_scale(const deviceSpinor<Ns,Nc> a, const deviceSpinor<Ns,Nc> b,
                   const val_t scalar) {
  constexpr auto rank = a.view.rank();
  const auto stream_array_size = a.view.extent(0);
  const Policy<rank> 
    default_policy(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}); 
  const auto tiling = get_tiling(a.view, default_policy);
  Kokkos::parallel_for(
      "scale",
      Policy<rank>(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}, 
                   tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                    const StreamIndex is, const StreamIndex ic)
      { 
        a.view(i,j,k,l,is,ic) = scalar * b.view(i,j,k,l,is,ic); 
      });
  Kokkos::fence();
}


template <int Ns, int Nc>
void perform_add(const deviceSpinor<Ns,Nc> a,
                 const deviceSpinor<Ns,Nc> b, deviceSpinor<Ns,Nc> c) {
  constexpr auto rank = a.view.rank();
  const auto stream_array_size = a.view.extent(0);
  const Policy<rank> 
    default_policy(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}); 
  const auto tiling = get_tiling(a.view, default_policy);
  Kokkos::parallel_for(
      "add",
      Policy<rank>(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}, 
                   tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                    const StreamIndex is, const StreamIndex ic)
      { 
        c.view(i,j,k,l,is,ic) = a.view(i,j,k,l,is,ic) + b.view(i,j,k,l,is,ic); 
      });
  Kokkos::fence();
}

template <int Ns, int Nc>
void perform_triad(const deviceSpinor<Ns,Nc> a, const deviceSpinor<Ns,Nc> b,
                   const deviceSpinor<Ns,Nc> c, const val_t scalar) {
  constexpr auto rank = a.view.rank();
  const auto stream_array_size = a.view.extent(0);
  const Policy<rank> 
    default_policy(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}); 
  const auto tiling = get_tiling(a.view, default_policy);
  Kokkos::parallel_for(
      "triad", 
      Policy<rank>(make_repeated_sequence<rank>(0), 
                   {stream_array_size,stream_array_size,stream_array_size,stream_array_size,Ns,Nc}, 
                   tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l,
                    const StreamIndex is, const StreamIndex ic)
      { 
        a.view(i,j,k,l,is,ic) = b.view(i,j,k,l,is,ic) + scalar * c.view(i,j,k,l,is,ic);
      });

  Kokkos::fence();
}

// int perform_validation(StreamHostArray &a, StreamHostArray &b,
//                        StreamHostArray &c, const StreamIndex arraySize,
//                        const val_t scalar) {
//   val_t ai = ainit;
//   val_t bi = binit;
//   val_t ci = cinit;
// 
//   for (StreamIndex i = 0; i < STREAM_NTIMES; ++i) {
//     ci = ai;
//     bi = scalar * ci;
//     ci = ai + bi;
//     ai = bi + scalar * ci;
//   };
// 
//   std::cout << "ai: " << ai << "\n";
//   std::cout << "a(0,0,0,0): " << a(0,0,0,0) << "\n";
//   std::cout << "bi: " << bi << "\n";
//   std::cout << "b(0,0,0,0): " << b(0,0,0,0) << "\n";
//   std::cout << "ci: " << ci << "\n";
//   std::cout << "c(0,0,0,0): " << c(0,0,0,0) << "\n";
//  
//   const double nelem = (double)arraySize*arraySize*arraySize*arraySize; 
//   const double epsilon = 2*4*STREAM_NTIMES*std::numeric_limits<val_t>::epsilon();
// 
//   double aError = 0.0;
//   double bError = 0.0;
//   double cError = 0.0;
// 
//   #pragma omp parallel reduction(+:aError,bError,cError)
//   {
//     double err = 0.0;
//     #pragma omp for collapse(2)
//     for (StreamIndex i = 0; i < arraySize; ++i) {
//       for (StreamIndex j = 0; j < arraySize; ++j) {
//         for (StreamIndex k = 0; k < arraySize; ++k) {
//           for (StreamIndex l = 0; l < arraySize; ++l) {
//             err = std::abs(a(i,j,k,l) - ai);
//             if( err > epsilon ){
//               //std::cout << "aError " << " i: " << i << " j: " << j << " k: " << k << " l: " << l << " err: " << err << "\n";
//               aError += err;
//             }
//             err = std::abs(b(i,j,k,l) - bi);
//             if( err > epsilon ){
//               //std::cout << "bError " << " i: " << i << " j: " << j << " k: " << k << " l: " << l << " err: " << err << "\n";
//               bError += err;
//             }
//             err = std::abs(c(i,j,k,l) - ci);
//             if( err > epsilon ){
//               //std::cout << "cError " << " i: " << i << " j: " << j << " k: " << k << " l: " << l << " err: " << err << "\n";
//               cError += err;
//             }
//           }
//         }
//       }
//     }
//   }
// 
//   std::cout << "aError = " << aError << "\n";
//   std::cout << "bError = " << bError << "\n";
//   std::cout << "cError = " << cError << "\n";
// 
//   val_t aAvgError = aError / nelem;
//   val_t bAvgError = bError / nelem;
//   val_t cAvgError = cError / nelem;
// 
//   std::cout << "aAvgErr = " << aAvgError << "\n";
//   std::cout << "bAvgError = " << bAvgError << "\n";
//   std::cout << "cAvgError = " << cAvgError << "\n";
// 
//   int errorCount       = 0;
// 
//   if (std::abs(aAvgError / ai) > epsilon) {
//     fprintf(stderr, "Error: validation check on View a failed.\n");
//     errorCount++;
//   }
// 
//   if (std::abs(bAvgError / bi) > epsilon) {
//     fprintf(stderr, "Error: validation check on View b failed.\n");
//     errorCount++;
//   }
// 
//   if (std::abs(cAvgError / ci) > epsilon) {
//     fprintf(stderr, "Error: validation check on View c failed.\n");
//     errorCount++;
//   }
// 
//   if (errorCount == 0) {
//     printf("All solutions checked and verified.\n");
//   }
// 
//   return errorCount;
// }

template <int Ns, int Nc>
int run_benchmark(const StreamIndex stream_array_size) {
  printf("Reports fastest timing per kernel\n");
  printf("Creating Views...\n");

  const double nelem = (double)stream_array_size*
                       (double)stream_array_size*
                       (double)stream_array_size*
                       (double)stream_array_size*
                       Ns*Nc;

  printf("Memory Sizes:\n");
  printf("- Array Size:    %" PRIu64 "^4\n",
         static_cast<uint64_t>(stream_array_size));
  printf("- Per Array:     %12.2f MB\n",
         1.0e-6 * nelem * (double)sizeof(val_t));
  printf("- Total: %12.2f MB\n",
         3.0e-6 * nelem * (double)sizeof(val_t));

  printf("Benchmark kernels will be performed for %d iterations.\n",
         STREAM_NTIMES);

  printf(HLINE);

  // WithoutInitializing to circumvent first touch bug on arm systems
  // StreamDeviceArray dev_a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "a"),
  //                         stream_array_size,stream_array_size,stream_array_size,stream_array_size);
  // StreamDeviceArray dev_b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "b"),
  //                         stream_array_size,stream_array_size,stream_array_size,stream_array_size);
  // StreamDeviceArray dev_c(Kokkos::view_alloc(Kokkos::WithoutInitializing, "c"),
  //                         stream_array_size,stream_array_size,stream_array_size,stream_array_size);

  // StreamHostArray a = Kokkos::create_mirror_view(dev_a);
  // StreamHostArray b = Kokkos::create_mirror_view(dev_b);
  // StreamHostArray c = Kokkos::create_mirror_view(dev_c);

  const val_t scalar(1.1);

  double setTime   = std::numeric_limits<double>::max();
  double copyTime  = std::numeric_limits<double>::max();
  double scaleTime = std::numeric_limits<double>::max();
  double addTime   = std::numeric_limits<double>::max();
  double triadTime = std::numeric_limits<double>::max();

  printf("Initializing Views...\n");

  deviceSpinor<Ns,Nc> dev_a(stream_array_size,stream_array_size,stream_array_size,stream_array_size,ainit);
  deviceSpinor<Ns,Nc> dev_b(stream_array_size,stream_array_size,stream_array_size,stream_array_size,binit);
  deviceSpinor<Ns,Nc> dev_c(stream_array_size,stream_array_size,stream_array_size,stream_array_size,cinit);

  printf("Starting benchmarking...\n");

  Kokkos::Timer timer;

  for (StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    perform_set(dev_c, 1.5);
    setTime = std::min(setTime, timer.seconds());

    timer.reset();
    perform_copy(dev_a, dev_c);
    copyTime = std::min(copyTime, timer.seconds());

    timer.reset();
    perform_scale(dev_b, dev_c, scalar);
    scaleTime = std::min(scaleTime, timer.seconds());

    timer.reset();
    perform_add(dev_a, dev_b, dev_c);
    addTime = std::min(addTime, timer.seconds());

    timer.reset();
    perform_triad(dev_a, dev_b, dev_c, scalar);
    triadTime = std::min(triadTime, timer.seconds());
  }

  // Kokkos::deep_copy(a, dev_a);
  // Kokkos::deep_copy(b, dev_b);
  // Kokkos::deep_copy(c, dev_c);

  // printf("Performing validation...\n");
  // int rc = perform_validation(a, b, c, stream_array_size, scalar);

  int rc = 0;

  printf(HLINE);

  printf("Set             %11.4f GB/s\n",
         1.0e-09 * 1.0 * (double)sizeof(val_t) * nelem / setTime);

  printf("Copy            %11.4f GB/s\n",
         1.0e-09 * 2.0 * (double)sizeof(val_t) * nelem / copyTime);
  
  printf("Scale           %11.4f GB/s\n",
         1.0e-09 * 2.0 * (double)sizeof(val_t) * nelem / scaleTime);
  
  printf("Add             %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * nelem / addTime);
  
  printf("Triad           %11.4f GB/s\n",
         1.0e-09 * 3.0 * (double)sizeof(val_t) * nelem / triadTime);

  printf(HLINE);

  return rc;
}

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("Kokkos 4D Spinor MDRangePolicy STREAM Benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  StreamIndex stream_array_size;
  rc = parse_args(argc, argv, stream_array_size);
  if (rc == 0) {
    rc = run_benchmark<4,3>(stream_array_size);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}
