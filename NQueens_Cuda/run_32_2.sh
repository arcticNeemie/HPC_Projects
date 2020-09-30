echo "Running CUDA tests using 32 threads per block and a CPU depth of 2"
echo "==================================================================" |& tee -a cuda_32_2.txt
./cuda -t 32 -d 2 5 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 6 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 7 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 8 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 9 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 10 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 11 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 12 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 13 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 14 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 15 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 16 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 17 |& tee -a cuda_32_2.txt;
./cuda -t 32 -d 2 18 |& tee -a cuda_32_2.txt;
