SRC=src/
BIN=bin/
CFLAGS=-w `pkg-config --cflags opencv` -O3 -fopenmp
LIBS=`pkg-config --libs opencv`

all: main

main: merge.o
	g++ $(CFLAGS) -o $(BIN)$@ $(BIN)$< $(SRC)main.cpp $(LIBS)

merge.o:
	g++ $(CFLAGS) -c -o $(BIN)merge.o $(SRC)merge.cpp $(LIBS)

run:
	time $(BIN)main -q 海洋

clean:
	rm -f $(BIN)*
