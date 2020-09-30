INC="./inc"
FLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc

all: KNN_serial KNN_parallel_sections KNN_tasks

KNN_serial: KNN_serial.c
	$(CC) KNN_serial.c -o KNN_serial -lm $(OMPFLAG)

KNN_parallel_sections: KNN_parallel_sections.c
	$(CC) KNN_parallel_sections.c -o KNN_parallel_sections -lm $(OMPFLAG)

KNN_tasks: KNN_tasks.c
	$(CC) KNN_tasks.c -o KNN_tasks -lm $(OMPFLAG)

clean:
	rm KNN_serial KNN_parallel_sections KNN_tasks
