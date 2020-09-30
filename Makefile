
CC	= /usr/local/cuda-10.0/bin/nvcc
LDFLAGS = -L /usr/local/cuda-10.0/lib64
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc

all:	simpleTexture serialParConv constant texture shared sharedConst #serialConv

serialParConv: serialParConv.cu
	$(CC) serialParConv.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) serialParConv.o  $(LDFLAGS) $(IFLAGS) -o serialParConv

sharedConst: sharedConst.cu
		$(CC) sharedConst.cu $(LDFLAGS) $(IFLAGS) -c $<
		$(CC) sharedConst.o  $(LDFLAGS) $(IFLAGS) -o sharedConst

texture: texture.cu
	$(CC) texture.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) texture.o  $(LDFLAGS) $(IFLAGS) -o texture

shared: shared.cu
	$(CC) shared.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) shared.o  $(LDFLAGS) $(IFLAGS) -o shared

constant: constant.cu
	$(CC) constant.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) constant.o  $(LDFLAGS) $(IFLAGS) -o constant

# serialConv: serialConv.cu
# 	$(CC) serialConv.cu $(LDFLAGS) $(IFLAGS) -c $<
# 	$(CC) serialConv.o  $(LDFLAGS) $(IFLAGS) -o serialConv

simpleTexture: simpleTexture.cu
	$(CC) simpleTexture.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) simpleTexture.o  $(LDFLAGS) $(IFLAGS) -o simpleTexture

clean:
	$(RM) serailParConv constant texture simpleTexture*.o *.~ shared sharedConst #serialConv
