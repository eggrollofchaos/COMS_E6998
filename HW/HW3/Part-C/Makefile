### Wei Alexander Xin - wax1
###
### Part-C Makefile for COMS E6998 HW3
### Insomnia CUDA 12.9
###

NVCC := /usr/local/cuda/bin/nvcc
OPTIONS := -O3

# WAX Note to graders:
# There were some issues with cuDNN install documented on Ed
# For cuDNN path, override if needed with CUDNN_PATH=/your/path
# Default assumes standard install; falls back to pip-installed location
CUDNN_PATH ?= /usr/local/cuda
CUDNN_FLAGS := -I$(CUDNN_PATH)/include -L$(CUDNN_PATH)/lib64 -L$(CUDNN_PATH)/lib -lcudnn

EXECS := c1 c2 c3
all: $(EXECS)

clean:
	rm -f $(EXECS)

c1: c1.cu
	$(NVCC) $< -o $@ $(OPTIONS)

c2: c2.cu
	$(NVCC) $< -o $@ $(OPTIONS)

# c3 requires cuDNN (pip-installed locally)
c3: c3.cu
	$(NVCC) $< -o $@ $(OPTIONS) $(CUDNN_FLAGS)
