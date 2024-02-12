CC = g++
CFLAGS = -std=c++11 -Wall -g -I./eigen3 

SRCS = 01_mnist_bonus.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = 01_mnist_bonus_program

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
