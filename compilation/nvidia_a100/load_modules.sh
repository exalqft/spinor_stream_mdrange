export MODULEPATH=/opt/software/easybuild-AMD/modules/all:/etc/modulefiles:/usr/share/modulefiles:/opt/software/modulefiles:/usr/share/modulefiles/Linux:/usr/share/modulefiles/Core:/usr/share/lmod/lmod/modulefiles/Core
module purge
module load \
  HDF5/1.14.0-gompi-2023a \
  UCX-CUDA/1.14.1-GCCcore-12.3.0-CUDA-12.1.1 \
  CMake/3.26.3-GCCcore-12.3.0 \
  OpenBLAS/0.3.23-GCC-12.3.0 \
  OpenMPI/4.1.5-GCC-12.3.0 \
  Boost/1.82.0-GCC-12.3.0

#  imkl/2023.2.0 \

