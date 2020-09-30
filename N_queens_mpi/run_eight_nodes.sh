echo "Running MPI Tests on eight nodes ... "
echo "===============================================" |& tee -a out_eight_node.txt
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 6 ./mpi 5 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 7 ./mpi 6 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 8 ./mpi 7 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 9 ./mpi 8 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 10 ./mpi 9 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 11 ./mpi 10 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 12 ./mpi 11 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 13 ./mpi 12 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 14 ./mpi 13 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 15 ./mpi 14 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 16 ./mpi 15 |& tee -a out_eight_node.txt;
mpiexec -hosts mscluster0,mscluster1,mscluster2,mscluster3,mscluster4,mscluster5,mscluster6,mscluster7 -n 17 ./mpi 16 |& tee -a out_eight_node.txt;
