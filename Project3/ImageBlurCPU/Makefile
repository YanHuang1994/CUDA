CC = gcc
CFLAGS = -Wall -g -O3 -std=c99

SRCS = ImageBlurCPU.c
OBJS = $(SRCS:.c=.o)
EXEC = ImageBlurCPU.out

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -lpthread -o $(EXEC)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
