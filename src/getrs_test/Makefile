SRC = $(wildcard ./*.cpp)
OBJ = $(SRC:.cpp=.o)
ASM = $(SRC:.cpp=.ll)
EXEC = getrs

CC = clang++
OPENMP = 
LDFLAGS  = -lm ${OPENMP}
CFLAGS = -std=c++14 -O3 -gline-tables-only -I ../FLENS -Wno-deprecated-declarations ${OPENMP}

all: $(EXEC)

%.o: %.cpp
	$(CC) -o $@ -c $(CFLAGS) $^

$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(OBJ) $(EXEC) $(ASM)
