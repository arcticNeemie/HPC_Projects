echo "Running CUDA tests using 64 threads per block and a CPU depth of 4"
echo "==================================================================" |& tee -a cuda_64_4.txt
./cuda -t 64 -d 4 5 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 6 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 7 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 8 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 9 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 10 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 11 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 12 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 13 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 14 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 15 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 16 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 17 |& tee -a cuda_64_4.txt;
./cuda -t 64 -d 4 18 |& tee -a cuda_64_4.txt;
