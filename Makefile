TARGET=main
OBJECTS=util.o mat_mul.o main.o

CXXFLAGS=-O3 -Wall
LDFLAGS=-lm -L/usr/local/cuda/lib64 -lcudart

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

mat_mul.o: mat_mul.cu
	nvcc -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	sbatch run.sh
