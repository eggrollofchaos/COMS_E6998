### Wei Alexander Xin - wax1
###
### Part-B Makefile for COMS E6998 HW3
### Insomnia CUDA 12.9
###

NVCC := /usr/local/cuda/bin/nvcc
CXX  := g++
OPTIONS := -O3

EXECS := q1 q2 q3
all: $(EXECS)

clean:
	rm -f $(EXECS)

q1: q1.cpp
	$(CXX) $< -o $@ $(OPTIONS)

q2: q2.cu
	$(NVCC) $< -o $@ $(OPTIONS)

q3: q3.cu
	$(NVCC) $< -o $@ $(OPTIONS)
