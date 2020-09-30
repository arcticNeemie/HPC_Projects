echo "Running MPI Tests on a single node ... "
echo "==================================================" |& tee -a out_single_node.txt
mpiexec -n 6 ./mpi 5 |& tee -a out_single_node.txt;
mpiexec -n 7 ./mpi 6 |& tee -a out_single_node.txt;
mpiexec -n 8 ./mpi 7 |& tee -a out_single_node.txt;
mpiexec -n 9 ./mpi 8 |& tee -a out_single_node.txt;
mpiexec -n 10 ./mpi 9 |& tee -a out_single_node.txt;
mpiexec -n 11 ./mpi 10 |& tee -a out_single_node.txt;
mpiexec -n 12 ./mpi 11 |& tee -a out_single_node.txt;
mpiexec -n 13 ./mpi 12 |& tee -a out_single_node.txt;
mpiexec -n 14 ./mpi 13 |& tee -a out_single_node.txt;
mpiexec -n 15 ./mpi 14 |& tee -a out_single_node.txt;
mpiexec -n 16 ./mpi 15 |& tee -a out_single_node.txt;
mpiexec -n 17 ./mpi 16 |& tee -a out_single_node.txt;
