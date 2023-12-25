HOME := $(shell dirname $$(dirname $$PWD))
INCLUDES := $(HOME)/include
PYBIND_INCLUDES := $$(python3 -m pybind11 --includes)

CFLAGS := -I$(INCLUDES) -g -fPIC -ffast-math -mfma -fopenmp -O3 -std=c++17 -ffunction-sections -ffast-math -mfma
LIBS := -L/home/szaman5/miniconda3/lib/ -lpython3 -lstdc++ -lm -lgomp -lquadmath -lpthread

all: mult_test

mult_test: test_matmul.cpp
	$(CXX) $^ -o test_matmul.out $(CFLAGS)
	./test_matmul.out
