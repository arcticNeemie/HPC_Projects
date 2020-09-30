echo "Running CUDA tests using 64 threads per block and a CPU depth of 5"
echo "==================================================================" |& tee -a cuda_64_5.txt
./cuda -t 64 -d 5 5 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 6 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 7 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 8 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 9 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 10 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 11 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 12 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 13 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 14 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 15 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 16 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 17 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 18 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 19 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 20 |& tee -a cuda_64_5.txt;
./cuda -t 64 -d 5 21 |& tee -a cuda_64_5.txt;
