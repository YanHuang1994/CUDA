CC = nvcc
CFLAGS = -std=c++11 -g -O2 -arch=sm_60

SRCS = ./src/ImageBlurGPU.cu
OBJS = $(SRCS:.cu=.o)
EXEC = ImageBlurGPU

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
