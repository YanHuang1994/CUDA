CC = gcc
CFLAGS = -Wall -g -O3 -std=c99

SRCS = CNN.c webgui.c
OBJS = $(SRCS:.c=.o)
EXEC = CNN.out

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -lm -lpthread -o $(EXEC)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
