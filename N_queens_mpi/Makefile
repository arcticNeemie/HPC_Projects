INC="../common/inc"
NVCCFLAGS=-I$(INC)
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall
MPIFLAGS = mpicc

all: mpi
mpi: N_queens_mpi.c
		$(MPIFLAGS) N_queens_mpi.c -o mpi
clean:
	rm mpi
