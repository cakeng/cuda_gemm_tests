TARGET=main
OBJECTS=util.o mat_mul.o main.o

CXXFLAGS=-O3 -Wall -fopenmp 
LDFLAGS=-lm -L/usr/local/cuda/lib64 -lcudart

ARCH= -gencode arch=compute_60,code=[sm_60,compute_60] \
      -gencode arch=compute_61,code=[sm_61,compute_61] \
	  -gencode arch=compute_75,code=[sm_75,compute_75] \
	  -gencode arch=compute_80,code=[sm_80,compute_80] \
      -gencode arch=compute_86,code=[sm_86,compute_86] \

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

mat_mul.o: mat_mul.cu
	nvcc $(ARCH) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	sbatch run.sh
