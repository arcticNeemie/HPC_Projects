INC="../common/inc"
NVCCFLAGS=-I$(INC) -I
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: cuda
cuda: N_queens_cuda_shared.cu
		$(NVCC) N_queens_cuda_shared.cu -o cuda
clean:
	rm cuda
