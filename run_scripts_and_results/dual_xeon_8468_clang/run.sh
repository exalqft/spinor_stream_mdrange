n4=(  4  8  12  16  24   32   48   64   80   96 )

echo nt n4 type kernel bw > results.dat 

# 64 threads, close binding -> use a single socket
# 128 threads, both sockets
for nt in 48 96; do

  export OMP_NUM_THREADS=$nt
  export OMP_PLACES=cores
  export OMP_PROC_BIND=close
  
  for n in $( seq 0 $(( ${#n4[@]} - 1 )) ); do
    for type in array-external array-internal static; do
      results=( $(./spinor-stream-mdrange-SC-${type} -n ${n4[$n]} | grep GB | awk '{print $1 " " $2}') )
      for i in $(seq 0 4); do
        echo $nt ${n4[$n]} ${type} ${results[$(( 2*$i ))]} ${results[$(( 2*$i + 1 ))]} | tee -a results.dat
      done
    done
  done || break
done
