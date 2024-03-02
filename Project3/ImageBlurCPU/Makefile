CC = g++
CFLAGS = -std=c++11 -Wall -g

SRCS = blurwindow.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC = blurwindow

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
