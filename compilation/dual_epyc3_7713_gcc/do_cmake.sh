source load_modules.sh 

CXXFLAGS="-O3 -mfma -mtune=znver3 -march=znver3" \
cmake \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DKokkos_ARCH_ZEN3=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  ~/code/spinor_stream_mdrange

